import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from tensorflow.python.data import Dataset
import numpy as np
import re
import sklearn.metrics as metrics
import pandas as pd
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
from collections import OrderedDict
tf.logging.set_verbosity(tf.logging.ERROR)
# Load data files
training_dataset = pd.read_csv('../input/train.csv', sep=',')
resources_dataset = pd.read_csv('../input/resources.csv', sep=',')
test_dataset = pd.read_csv('../input/test.csv', sep=',')
# Join train and test data. Context features are based on the full dataset 
dfall = training_dataset[["id","teacher_id","project_submitted_datetime","project_is_approved"]].append(
            test_dataset[["id","teacher_id","project_submitted_datetime"]])

# Parse project submitted date and sort the data frame
dfall["project_submitted_datetime"] = pd.to_datetime(dfall["project_submitted_datetime"])
dfall = dfall.sort_values(by=["project_submitted_datetime"])
dfall = dfall.set_index("project_submitted_datetime")

# Calculate rolling 1 day approval rate feature (roll_approved_pct)
#  - Calculated as approved applications within the last 24 hours / total applications within the last 24 hours
#  - Always exclude the current application from calculation
dfall["project_is_approved1"] = dfall["project_is_approved"].fillna(0)
dfall["train_data"] =1-( dfall["project_is_approved"].isna()*1)
dfall[["roll_approved","roll_total"]] = dfall.rolling('1d')["project_is_approved1","train_data"].sum()
dfall["roll_approved_pct"] = (dfall["roll_approved"]-dfall["project_is_approved1"])/(dfall["roll_total"]-dfall["train_data"])
dfall = dfall.reset_index(level="project_submitted_datetime")

# Create teacher-context features
#  - Sort by teacher ID + project submitted datetime and shift forward and backward to get information about next/previous application
dfall = dfall.sort_values(by=["teacher_id","project_submitted_datetime"])
dfall[["last_project_is_approved","last_dt"]] = dfall.groupby("teacher_id")["project_is_approved","project_submitted_datetime"].shift(1)
dfall[["next_project_is_approved","next_dt"]] = dfall.groupby("teacher_id")["project_is_approved","project_submitted_datetime"].shift(-1)
dfall["last_project_is_rejected"] = 1-dfall["last_project_is_approved"]
dfall["next_project_is_rejected"] = 1-dfall["next_project_is_approved"]
dfall["time_since_last_project"] = (dfall["project_submitted_datetime"] - dfall["last_dt"])/np.timedelta64(1, 'h')
dfall["time_to_next_project"] = (dfall["next_dt"] - dfall["project_submitted_datetime"])/np.timedelta64(1, 'h')
dfall["first_project_ind"] = dfall["time_since_last_project"].isna()*1
dfall["last_project_ind"] = dfall["time_to_next_project"].isna()*1
#  Use rank to get the position of the application in teacher's submission history
dfall["project_number"] = dfall.groupby("teacher_id")["project_submitted_datetime"].rank().astype(int)

#  Clean up variables
dfall = dfall.fillna({"last_project_is_approved":0,"next_project_is_approved":0,
              "last_project_is_rejected":0,"next_project_is_rejected":0,})
for col in ["last_project_is_approved","next_project_is_approved","last_project_is_rejected","next_project_is_rejected"]:
    dfall[col] = dfall[col].fillna(0)
    dfall[col] = dfall[col].astype(int)
dfall.head(5)
# Split training and validation data
N_TRAINING = 60000
N_VALIDATION = 20000 

# Create separate training and validation datasets
training_dataset = training_dataset.reindex(np.random.RandomState(seed=67).permutation(training_dataset.index))
training_data = training_dataset.head(N_TRAINING).copy()
validation_data = training_dataset.tail(N_VALIDATION).copy()
# Impute missing descriptions 
resources_dataset = resources_dataset.fillna({'description':'N/A'})
# Calculate total price for each resource
resources_dataset['total_price'] = resources_dataset['quantity'] * resources_dataset['price']
# Aggregate resources on id level: count rows, add total price, append descriptions
grp = resources_dataset.groupby(['id'])
resources_dataset_grp = grp.apply(lambda row: pd.Series(dict(
    resource_count=row['total_price'].count(),
    resource_price=row['total_price'].sum(),
    resource_descriptions=' '.join(row['description']) )) ).reset_index()
resources_dataset_grp.head(5)
# Calculation of teacher statistics is based on all training data
training_data_ext = training_dataset[["teacher_id","project_submitted_datetime","project_is_approved"]].copy()

# Discretize submitted datetime on day and month level
for ds in [training_data,validation_data,test_dataset,training_data_ext]:
    ds["project_submitted_date"] = pd.to_datetime(ds["project_submitted_datetime"]).dt.date
    ds["project_submitted_month"] = pd.to_datetime(ds["project_submitted_datetime"]).dt.to_period('M')

# Caluclate all-time teacher stats
teacher_stats = pd.DataFrame(training_data_ext.groupby("teacher_id")["project_is_approved"].agg(["count","sum"]).reset_index())
teacher_stats = teacher_stats.rename(index=str,columns={"count":"app_cnt","sum":"approved_cnt"})
teacher_stats["approved_pct"] = teacher_stats["approved_cnt"]/teacher_stats["app_cnt"]
teacher_stats["rejected_cnt"] =teacher_stats["app_cnt"] - teacher_stats["approved_cnt"]
training_data = pd.merge(training_data, teacher_stats, on='teacher_id', how="left")
validation_data = pd.merge(validation_data, teacher_stats, on='teacher_id', how="left")
test_dataset = pd.merge(test_dataset, teacher_stats, on='teacher_id', how="left")

# Caluclate teacher stats for each day and month
for period in ["date","month"]:
    teacher_stats = pd.DataFrame(training_data_ext.groupby(["teacher_id","project_submitted_" + period])["project_is_approved"].agg(["count","sum"]).reset_index())
    teacher_stats = teacher_stats.rename(index=str,columns={"count":"same_" + period + "_app_cnt","sum":"same_" + period + "_approved_cnt"})
    teacher_stats["same_" + period + "_approved_pct"] = teacher_stats["same_" + period + "_approved_cnt"]/teacher_stats["same_" + period + "_app_cnt"]
    teacher_stats["same_" + period + "_rejected_cnt"] = teacher_stats["same_" + period + "_app_cnt"] - teacher_stats["same_" + period + "_approved_cnt"]
    training_data = pd.merge(training_data, teacher_stats, on=['teacher_id','project_submitted_' + period], how="left")
    validation_data = pd.merge(validation_data, teacher_stats, on=['teacher_id','project_submitted_' + period], how="left")
    test_dataset = pd.merge(test_dataset, teacher_stats, on=['teacher_id','project_submitted_' + period], how="left")

# Inpute zeroes for all missing counts
fillNaDict = {"app_cnt":0,"approved_cnt":0,"rejected_cnt":0,
              "same_date_app_cnt":0,"same_date_approved_cnt":0,"same_date_rejected_cnt":0,
              "same_month_app_cnt":0,"same_month_approved_cnt":0,"same_month_rejected_cnt":0}
training_data = training_data.fillna(fillNaDict)
validation_data = validation_data.fillna(fillNaDict)
test_dataset = test_dataset.fillna(fillNaDict)

# Adjust all stats, so that current row (and its approval) is excluded from the calculation
for ds in [training_data,validation_data]:
    for prd in ["","same_date_","same_month_"]:
        ds[prd + "app_cnt"] = ds[prd + "app_cnt"] - 1
        ds[prd + "approved_cnt"] = ds[prd + "approved_cnt"] - ds["project_is_approved"]
        ds[prd + "rejected_cnt"] = ds[prd + "app_cnt"] - ds[prd + "approved_cnt"]
        ds[prd + "approved_pct"] = ds[prd + "approved_cnt"]/ds[prd + "app_cnt"]

training_data.head(5)
# Merge grouped resource data to the training and test datasets 
training_data = pd.merge(training_data, resources_dataset_grp, on='id', how="left")
validation_data = pd.merge(validation_data, resources_dataset_grp, on='id', how="left")
test_dataset = pd.merge(test_dataset, resources_dataset_grp, on='id', how="left")

# Merge time-context resource data to the training and test datasets 
dropList = ["teacher_id","project_submitted_datetime","project_is_approved"]
training_data = pd.merge(training_data, dfall.drop(columns=dropList), on='id', how="left")
validation_data = pd.merge(validation_data, dfall.drop(columns=dropList), on='id', how="left")
test_dataset = pd.merge(test_dataset, dfall.drop(columns=dropList), on='id', how="left")
training_data.head(5)
datasets = [training_data,validation_data,test_dataset]
# 1. Multi-Category Columns
#   These columns include arbitrary number of comma-separated key phrases.
#    ->  We will remove spaces and then replace commas with spaces. That will allow us to treat the data as text with each key phrase being one "word"
multi_cat_cols = ["project_subject_categories","project_subject_subcategories"]
for col in multi_cat_cols:
    for ds in datasets:
        ds[col] = ds[col].str.replace(" ","").str.replace(","," ")

# 2. Text columns
#  - Clean: Remove punctuation, convert to lower case
#  - Engineer features: word_count, is_na
text_cols = ["project_title","project_essay_1","project_essay_2", #"project_essay_3","project_essay_4",
             "project_resource_summary","project_subject_categories","project_subject_subcategories","resource_descriptions"]
max_word_count = 500
for col in text_cols:
    for ds in datasets:
        ds[col + "_is_na"] = ds[col].isnull() * 1
        ds[col] = ds[col].fillna('').str.lower().str.replace('[^\w\s]','')
        ds[col + "_word_count"] = ds[col].str.count(' ') + 1
        ds[col] = ds[col].str.split(' ',max_word_count).str[0:max_word_count].str.join(' ')

# Impute missing data for the teacher_prefix feature
for ds in datasets:
    ds["teacher_prefix"] = ds["teacher_prefix"].fillna('')

# Project Submitted Datetime Features
for ds in datasets:
    ds["project_submitted_datetime_dt"] = pd.to_datetime(ds["project_submitted_datetime"])
    ds["submitted_year"] = ds["project_submitted_datetime_dt"].dt.year
    ds["submitted_month"] = ds["project_submitted_datetime_dt"].dt.month
    ds["submitted_dow"] = ds["project_submitted_datetime_dt"].dt.weekday_name
    ds["submitted_dom"] = ds["project_submitted_datetime_dt"].dt.day

training_data.head(5)
# Define lists of all feature types
numeric_features = ["app_cnt","approved_cnt","rejected_cnt",
                    "same_date_app_cnt","same_date_approved_cnt","same_date_rejected_cnt",
                    "same_month_app_cnt","same_month_approved_cnt","same_month_rejected_cnt"]
numeric_features_bucket = ["teacher_number_of_previously_posted_projects","resource_price",
                    "time_since_last_project","time_to_next_project"
                    ,"approved_pct","same_date_approved_pct","same_month_approved_pct","project_number",
                    "roll_approved_pct"] + [col+"_word_count" for col in text_cols]
# Custom cutoffs for selected bucketized numeric features
numeric_features_bucket_cutoffs = {}
numeric_features_bucket_cutoffs["time_since_last_project"] = [0.1,0.2,0.3,0.5,1.0,6.0,24.0,168.0,336.0,720.0,8760.0]
numeric_features_bucket_cutoffs["time_to_next_project"] = numeric_features_bucket_cutoffs["time_since_last_project"]
numeric_features_bucket_cutoffs["project_number"] = [1.0,2.0,3.0,4.0,10.0,20.0]

categorical_features = ["project_grade_category","teacher_prefix","submitted_year","submitted_month","submitted_dow"]
categorical_features_embed = ["school_state","submitted_dom"]
binary_features = ["last_project_is_approved","next_project_is_approved",
                   "last_project_is_rejected","next_project_is_rejected",
                   "first_project_ind", "last_project_ind"] + [col + "_is_na" for col in text_cols]
text_features = text_cols
target = "project_is_approved"

# Based on the lists of column names above, create list of TensorFlow features 
features = []
for feature in binary_features:
    features.append( tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_identity(feature, 2) ))

for feature in numeric_features_bucket:
    quantiles = []
    if feature in numeric_features_bucket_cutoffs:
        quantiles = numeric_features_bucket_cutoffs[feature]
    else:
        num_buckets = 20
        quantiles = training_data[feature].quantile(np.arange(1.0, num_buckets) / num_buckets)
        quantiles = [quantiles[q] for q in quantiles.keys()]
        quantiles = list(OrderedDict.fromkeys(quantiles))
    #print(feature)
    #print(quantiles)
    features.append( tf.feature_column.bucketized_column(tf.feature_column.numeric_column(feature), boundaries=quantiles) )

for feature in numeric_features:
    features.append( tf.feature_column.numeric_column(feature) )

for feature in categorical_features:
    dictionary = training_data[feature].unique()
    features.append(tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list(feature,dictionary)))

for feature in categorical_features_embed:
    embedding_count = 2
    dictionary = training_data[feature].unique()
    features.append(tf.feature_column.embedding_column(tf.feature_column.categorical_column_with_vocabulary_list(feature,dictionary),embedding_count))

# Helper function for text feature definition: Create dictionary of significant terms (return a list)
def getSignificantTerms(feature,posCountThreshold,wordCount):
    # Count term frequencies among accepted and among rejected applications
    tokenizerPos = Tokenizer()
    tokenizerNeg = Tokenizer()
    tokenizerPos.fit_on_texts(training_data.loc[training_data[target]==1][feature])
    tokenizerNeg.fit_on_texts(training_data.loc[training_data[target]==0][feature])
    posCounts = tokenizerPos.word_counts
    negCounts = tokenizerNeg.word_counts
    words = []
    # Iterate over all terms in the accepted submissions
    for w in posCounts:
        p = posCounts[w]
        # Term must appear at least posCountThreshold-times in the accepted submissions to be considered
        if p > posCountThreshold:
            n = 0
            if w in negCounts:
                n = negCounts[w]
            # Add term to candidate list with some basic stats
            # - word, freq
            words.append([w,p,n,p/(p+n)])    
    if len(words) <= wordCount:
        return [x[0] for x in words]
    words = sorted(words,key=lambda x: -x[3])
    wordCountPart = int(wordCount/2)
    ret = [x[0] for x in words[0:wordCountPart]]  # Most overrepresentd in accepted
    ret += ret + [x[0] for x in words[-wordCountPart:]]  # Most underrepresented in accepted
    return set(ret)

# Helper function for text feature definition: Prune text fields to keep only dictionary words
def dropNonDictionaryWords(feature,dictionary):
    dict_set = set(dictionary)
    for ds in [training_data,validation_data,test_dataset]:
        ds[feature] = ds[feature].apply(lambda x: set([y for y in x.split(' ') if y in dict_set])).str.join(' ')
    #print(training_data[feature][:10])
    
for feature in text_features:
    max_words = 500
    min_pos_frequency = 200
    embedding_count = 4
    dictionary = getSignificantTerms(feature,min_pos_frequency,max_words)
    #print(feature)
    dropNonDictionaryWords(feature,dictionary)
    features.append(tf.feature_column.embedding_column(tf.feature_column.categorical_column_with_vocabulary_list(feature,dictionary),embedding_count))
# Helper function for text feature ipnut: Translate text (string with words) into a list of words 
def _parse_text(features,targets):
    for key in text_features:
        features[key] = tf.string_split([features[key]]).values
        #print(features[key])
    return (features,targets)
    
def my_input_fn( data, batch_size=1, shuffle=True, num_epochs=None ):
    # Create dictionary of all columns that are used by the features as defined above
    features = {key:np.array(data[key]) for key in (binary_features + numeric_features + numeric_features_bucket + categorical_features 
                                                    + categorical_features_embed + text_features)}
    targets = data[target]
    
    ds = Dataset.from_tensor_slices((features,targets)) # warning: 2GB limit
    ds = ds.map(_parse_text)
    ds = ds.padded_batch(batch_size, ds.output_shapes).repeat(num_epochs)
    # Shuffle the data, if specified
    if shuffle:
      ds = ds.shuffle(10000)
    
    # Return the next batch of data
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels
# Define Estimator
estimator = tf.estimator.DNNClassifier(
    feature_columns=features,
    hidden_units=[10,10],
    optimizer=tf.train.ProximalAdagradOptimizer(
      learning_rate=0.1,
      l1_regularization_strength=0.0001
    ))
# Create training and scoring versions of the input function 
train_fn = lambda:my_input_fn(data=training_data,batch_size=100)
trainev_fn = lambda:my_input_fn(data=training_data,num_epochs=1,shuffle=False)
valid_fn = lambda:my_input_fn(data=validation_data,num_epochs=1,shuffle=False)
# Train model
estimator.train(input_fn=train_fn, steps=2000)
# Calculate training and validation metrics
training_metrics = estimator.evaluate(input_fn=trainev_fn)
validation_metrics = estimator.evaluate(input_fn=valid_fn)
print("AUC train/test: {}/{}".format(training_metrics['auc'],validation_metrics['auc']))
# Make predictions
test_dataset["project_is_approved"] = 0
test_fn = lambda:my_input_fn(data=test_dataset,num_epochs=1,shuffle=False)
predictions_generator = estimator.predict(input_fn=test_fn)
predictions_list = list(predictions_generator)

# Extract probabilities
probabilities = [p["probabilities"][1] for p in predictions_list]

my_submission = pd.DataFrame({'id': test_dataset["id"], 'project_is_approved': probabilities})

my_submission.to_csv('my_submission.csv', index=False)