# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
print(os.listdir("/kaggle/input/petfinder-adoption-prediction"))
print(os.listdir("/kaggle/input/petfinder-adoption-prediction/train"))
trainData = pd.read_csv('/kaggle/input/petfinder-adoption-prediction/train/train.csv')
trainData.head()
trainData.info()
#Adoption speed
print(trainData['AdoptionSpeed'].value_counts())
trainData['AdoptionSpeed'].hist(grid = False)
# Type 1 = dog
# Type 2 = cat
trainData['Type'].value_counts()
# Adoption speed for dogs
trainData[trainData['Type'] == 1]['AdoptionSpeed'].hist(grid=False);
# Adoption speed for cats
trainData[trainData['Type'] == 2]['AdoptionSpeed'].hist(grid=False);
for w in trainData['Name']:
    print(w)
#Special names: 
#1) nan
#2) Multiple names: Siu Pak & Her 6 Puppies
#3) Generic names: 2 Mths Old Cute Kitties, Lost Dog, No Name, 9 Puppies For Adoption!, No Name Yet, Not Named, (No Name),[No Name], etc.
#4) Meaningless names: IO-Male-03, H3, Z3, DO RE MI, BB, Y1
for name in trainData[trainData['Name'].isna()]['Name']:
    print(name)
names = []
for name in trainData['Name']:
    if 'name' in str(name).lower():
        names.append(str(name))
   
for name in names:
    print(name)
    
# No names:
# Lost Dog, No Name, 9 Puppies For Adoption!, No Name Yet, Not Named, (No Name),[No Name], $ To Be Named $, Noname, Unamed Yet 2, Unamed, Unnamed, No Names Yet, Not Named Yet,
# Unnamed 3 Kittens ( By Dani), No Name Kitten, Nameless, (no Name), Name Them & Love Them, Not Name Yet, No Names Yet, *No Name*, "no Name", (No Names Yet), * To Be Named *,
# Unnamed., NO NAME, Not Yet Name, No Name Kitties, Waiting For You To Give Him A Name, No Names Yet, *please Name Us*, Newborn *no Name, - To Be Named -, 
# No Name Yet, It's Up To The Owner, Name Them & Love Them 3, NO NAME YET, (No Name - She Is Just A Stray), Cream Cat (unnamed), (no Name), Wait For The Real Owner To Name It,
# 4 Kittens Open For Adoption (no Name), Need You Giving  A Name, No Name 2, UNNAMED, Unamed Yet, No Name Yet...., Kitten....no Name, Name Less Kitten, Haven't Named Them,
# No Name Yet (Must Neuter), Haven't Name Yet, Haven't Been Named, Not Yet Named
# Normalizing "nan" names to empty strings.
trainData.loc[trainData['Name'].isna(), 'Name'] = ""
print("Number of 'NaN' names: " + str(len(trainData[trainData['Name'] == ""])))
# Normalizing different forms of "Unnamed"
unnamedForms = set(['Lost Dog', 'No Name', '9 Puppies For Adoption!', 'No Name Yet', 'Not Named', '(No Name)', '[No Name]', '$ To Be Named $', 'Noname', 'Unamed Yet 2',\
               'Unamed', 'Unnamed', 'No Names Yet', 'Not Named Yet', 'Unnamed 3 Kittens ( By Dani)', 'No Name Kitten', 'Nameless', '(no Name)', 'Name Them & Love Them', \
               'Not Name Yet', 'No Names Yet', '*No Name*', '"no Name"', '(No Names Yet)', '* To Be Named *', 'Unnamed.', 'NO NAME', 'Not Yet Name', 'No Name Kitties', \
               'Waiting For You To Give Him A Name', 'No Names Yet', '*please Name Us*', 'Newborn *no Name', '- To Be Named -', 'No Name Yet, It\'s Up To The Owner', \
               'Name Them & Love Them 3', 'NO NAME YET', '(No Name - She Is Just A Stray)', 'Cream Cat (unnamed)', '(no Name)', 'Wait For The Real Owner To Name It', \
               '4 Kittens Open For Adoption (no Name)', 'Need You Giving  A Name', 'No Name 2', 'UNNAMED', 'Unamed Yet', 'No Name Yet....', 'Kitten....no Name', \
               'Name Less Kitten', 'Haven\'t Named Them', 'No Name Yet (Must Neuter)', 'Haven\'t Name Yet', 'Haven\'t Been Named', 'Not Yet Named'])

trainData.loc[trainData['Name'].isin(unnamedForms), 'Name'] = ""
trainData[trainData['Name'].isin(unnamedForms)] = ""
print("Number of 'NaN' names: " + str(len(trainData[trainData['Name']==""])))
#print(trainData[trainData['Name'] == ""])
# Removing names that are codes (no vowels or two characters or less)
codeNames = set()
for name in trainData['Name']:
    strName = str(name).lower()
    if len(strName) < 3 or ('a' not in strName and 'e' not in strName and 'i' not in strName and 'o' not in strName and 'u' not in strName and 'y' not in strName):
        codeNames.add(strName)
    
print("Found " + str(len(names)) + " code names")

trainData.loc[trainData['Name'].isin(codeNames), 'Name'] = ""
print("Number of 'NaN' names: " + str(len(trainData[trainData['Name']==""])))
# Adding feature for length of name.
trainData['Name_Length'] = trainData['Name'].map(str).apply(len)
print(trainData['Name_Length'].value_counts())
# Adoption speed breakdown percentage for unnamed pets
100*(trainData[trainData['Name'] == ""]["AdoptionSpeed"]).value_counts() / (len(trainData[trainData['Name'] == ""]))
# Adoption speed breakdown percentage for named pets
100*(trainData[trainData['Name'] != ""]["AdoptionSpeed"]).value_counts() / (len(trainData[trainData['Name'] != ""]))
# Unnamed pets tend to be unadopted for more than 100 days by 7%, in comparison with named pets.
#trainData[["Age", "AdoptionSpeed"]].hist()
trainData.boxplot(column=['Age'], by=['AdoptionSpeed'])
trainData[trainData["AdoptionSpeed"] == 0]["Age"].value_counts()
# Adding feature of Age in Years to group together pets with similar age.
trainData['Age_Years'] = trainData['Age'] // 12
trainData['Age_Years'].value_counts()
trainData.boxplot(column=['Age_Years'], by=['AdoptionSpeed'])
# It seems that younger pets are preferred: Pets adopted the same day are exclusively under 10 years old, while pets over 15 years old take more than 30
# days to get adopted.
# Breed Data
breedData = pd.read_csv('/kaggle/input/petfinder-adoption-prediction/breed_labels.csv')
breedData.head(10)
# Creating feature "IsPureBreed"
trainData['IsPureBreed'] = (trainData["Breed1"] == 0) | (trainData["Breed2"] == 0) | (trainData["Breed1"] == trainData["Breed2"])
#100*(trainData[trainData['Name'] != ""]["AdoptionSpeed"]).value_counts() / (len(trainData[trainData['Name'] != ""]))
print(100*trainData[trainData['IsPureBreed']]["AdoptionSpeed"].value_counts() / (len(trainData[trainData['IsPureBreed']])))
trainData[trainData['IsPureBreed']]["AdoptionSpeed"].hist()
print(100*trainData[~trainData['IsPureBreed']]["AdoptionSpeed"].value_counts() / (len(trainData[~trainData['IsPureBreed']])))
trainData[~trainData['IsPureBreed']]["AdoptionSpeed"].hist()
# Interestingly, mixed breeds have better adoption rates for the same day, 
# and much better chances of getting adopted before 100 days (only 19% take more than 100 days, versus 29% for purebreeds)
#Health
trainData["Health"].value_counts()
# Adoption speed for healthy animals
print(100*trainData[trainData['Health'] == 1]['AdoptionSpeed'].value_counts() / (len(trainData[trainData['Health'] == 1])))
trainData[trainData['Health'] == 1]['AdoptionSpeed'].hist(grid=False);
# Adoption speed for animals with minor injuries
print(100*trainData[trainData['Health'] == 2]['AdoptionSpeed'].value_counts() / (len(trainData[trainData['Health'] == 2])))
trainData[trainData['Health'] == 2]['AdoptionSpeed'].hist(grid=False);
# Adoption speed for animals with serious injuries
print(100*trainData[trainData['Health'] == 3]['AdoptionSpeed'].value_counts() / (len(trainData[trainData['Health'] == 3])))
trainData[trainData['Health'] == 3]['AdoptionSpeed'].hist(grid=False)
# Healthy animals tend to get adopted more easily than animals with minor injuries and with series injuries. 
# In particular, 27% of healthy animals are not adopted within 100 days, while animals with minor injures have a rate of 35%, 
# and animals with serious injuries have a rate of 41% of not getting adopted after 100 days.
# Photo amount.
trainData["PhotoAmt"].value_counts()
print(100*trainData[trainData['PhotoAmt'] == 0]['AdoptionSpeed'].value_counts() / (len(trainData[trainData['PhotoAmt'] == 0])))
trainData[trainData['PhotoAmt'] == 0]['AdoptionSpeed'].hist()
print(100*trainData[trainData['PhotoAmt'] == 1]['AdoptionSpeed'].value_counts() / (len(trainData[trainData['PhotoAmt'] == 1])))
trainData[trainData['PhotoAmt'] == 1]['AdoptionSpeed'].hist()
print(100*trainData[trainData['PhotoAmt'] == 2]['AdoptionSpeed'].value_counts() / (len(trainData[trainData['PhotoAmt'] == 2])))
trainData[trainData['PhotoAmt'] == 2]['AdoptionSpeed'].hist()
print(100*trainData[trainData['PhotoAmt'] > 2]['AdoptionSpeed'].value_counts() / (len(trainData[trainData['PhotoAmt'] > 2])))
trainData[trainData['PhotoAmt'] > 2]['AdoptionSpeed'].hist()
# The percentage of pets not adopted after 100 days for profiles with zero photos is 64%, while one or more photos seem to increase the chances of 
# being adopted earlier.
# Creating feature "HasPhoto"
trainData['HasPhoto'] = trainData["PhotoAmt"] > 0
# VideoAmt feature
trainData['VideoAmt'].value_counts()
# The VideoAmt category is very skewed to zero, as there rarely are any videos attahced to the profiles, so it's hard to make an inference with such a class imbalance.
# Quantity feature.
trainData['Quantity'].value_counts()
# The Quantity category is also strongly skewed to profiles with one animal.

# Sentiment Analysis
# Create dataframe for Sentiment Analysis
import json

petIds = []
magnitudes = []
scores = []

for sentimentFilename in os.listdir("/kaggle/input/petfinder-adoption-prediction/train_sentiment"):
    with open("/kaggle/input/petfinder-adoption-prediction/train_sentiment/" + sentimentFilename, 'r') as f:
        jsonContent = json.loads(f.read())
        magnitude = jsonContent['documentSentiment']['magnitude']
        score = jsonContent['documentSentiment']['score']
        petIds.append(sentimentFilename.split('.')[0])
        magnitudes.append(float(magnitude))
        scores.append(float(score))
rows = []
for i in range(len(petIds)):
    rows.append([petIds[i], magnitudes[i], scores[i]])
columns = ['PetID', 'SentimentMagnitude', 'SentimentScore']

sentimentDf=pd.DataFrame(rows)
sentimentDf.columns = columns

sentimentDf.head(10)
trainDataWithSentiment = pd.merge(trainData, sentimentDf, on='PetID')
trainDataWithSentiment.head(100)
# Sentiment Analysis
trainDataWithSentiment['SentimentScore'].value_counts()
#print(100*trainDataWithSentiment[trainDataWithSentiment['SentimentScore'] > 0]['AdoptionSpeed'].value_counts() / (len(trainDataWithSentiment[trainDataWithSentiment['SentimentScore'] > 0])))
#trainData[trainDataWithSentiment['SentimentScore'] > 0]['AdoptionSpeed'].hist()
trainDataWithSentiment[trainDataWithSentiment['SentimentScore'] > 0]['AdoptionSpeed'].hist()
trainDataWithSentiment[trainDataWithSentiment['SentimentScore'] < 0]['AdoptionSpeed'].hist()
trainDataWithSentiment.boxplot(column=['SentimentScore'], by=['AdoptionSpeed'])
trainDataWithSentiment.boxplot(column=['SentimentMagnitude'], by=['AdoptionSpeed'])
trainDataWithSentiment['SentimentMultiplier'] = trainDataWithSentiment['SentimentScore'] * trainDataWithSentiment['SentimentMagnitude']
trainDataWithSentiment.boxplot(column=['SentimentScore'], by=['AdoptionSpeed'])
trainDataWithSentiment.boxplot(column=['SentimentMagnitude'], by=['AdoptionSpeed'])
trainDataWithSentiment.boxplot(column=['SentimentMultiplier'], by=['AdoptionSpeed'])
# As the sentiment score becomes more negative adoption takes longer or becomes more unlikely.
# Larger sentiment magnitudes also results in the pets taking longer to be adopted.
# As the sentimient multipliers (score * magnitude) become more extreme (positively or negatively) the pet will take longer to get adopted.
# Adding DescriptionLength
trainDataWithSentiment['DescriptionLength'] = trainDataWithSentiment['Description'].map(str).apply(len)
print(trainDataWithSentiment['DescriptionLength'].value_counts())
# Adding IsSinglePet
trainDataWithSentiment['IsSinglePet'] = trainDataWithSentiment['Quantity'] == 1
print(trainDataWithSentiment['IsSinglePet'].value_counts())
# Adding IsFree
trainDataWithSentiment['IsFree'] = trainDataWithSentiment['Fee'] == 0
print(trainDataWithSentiment['IsFree'].value_counts())
# Adding HasVideo
trainDataWithSentiment['HasVideo'] = trainDataWithSentiment['VideoAmt'] > 0
print(trainDataWithSentiment['HasVideo'].value_counts())
# Building a model

#trainColumns = ['Type', 'Name', 'Age', 'Breed1', 'Breed2', 'Health', 'PhotoAmt', 'SentimentScore', 'SentimentMagnitude', \
#            'Name_Length', 'Age_Years', 'IsPureBreed', 'HasPhoto', 'SentimentMultiplier', 'AdoptionSpeed']

#trainColumns_lgb = ['Type', 'Age', 'Breed1', 'Breed2', 'Health', 'PhotoAmt', 'SentimentScore', 'SentimentMagnitude', \
#                    'Name_Length', 'Age_Years', 'IsPureBreed', 'HasPhoto', 'SentimentMultiplier', 'AdoptionSpeed']


trainColumns_lgb = ['Type', 'Age', 'Breed1', 'Breed2', 'Gender', 'Color1', 'Color2', \
       'Color3', 'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed', \
       'Sterilized', 'Health', 'Fee', 'SentimentScore', 'SentimentMagnitude', \
       'Name_Length', 'Age_Years', 'IsPureBreed', 'HasPhoto', 'SentimentMultiplier', \
       'PhotoAmt', 'DescriptionLength', 'Quantity', 'IsSinglePet', 'IsFree', 'VideoAmt', 'HasVideo', 'AdoptionSpeed']

#trainColumns_lgb = ['Type', 'Age', 'Breed1', 'Breed2', 'Health', 'SentimentScore', 'SentimentMagnitude', \
#       'Name_Length', 'Age_Years', 'IsPureBreed', 'HasPhoto', 'SentimentMultiplier', \
#       'PhotoAmt', 'AdoptionSpeed']


#lgb_categorical_features = ['Type', 'Breed1', 'Breed2', 'Health']

lgb_categorical_features = ['Type', 'Breed1', 'Breed2', 'Gender', 'Color1', 'Color2', 'Color3', 'Health', 'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed', \
                           'Sterilized']

lgb_bool_features = ['HasPhoto', 'IsSinglePet', 'IsFree', 'IsPureBreed', 'HasVideo']


#lgb_categorical_features = ['Type', 'Breed1', 'Breed2', 'Health', 'HasPhoto']

#lgb_numerical_features = ['Age', 'PhotoAmt', 'Age_Years']
lgb_numerical_features = ['Age', 'PhotoAmt', 'Age_Years', 'Fee', 'SentimentScore', 'SentimentMagnitude', 'Name_Length', 'SentimentMultiplier', \
                          'DescriptionLength', 'Quantity', 'VideoAmt']
#lgb_numerical_features = ['Age', 'PhotoAmt', 'Age_Years', 'SentimentScore', 'SentimentMagnitude', 'Name_Length', 'SentimentMultiplier']

# Type, Name, Age, Breed1, Breed2, Health, PhotoAmt, Age_Years

X_train_lgb = trainDataWithSentiment[trainColumns_lgb].copy()
# Change this
X_test_lgb = X_train_lgb.copy()
def train_models(X_train, X_test, categorical_features, numerical_features, bool_features):
    
    import lightgbm as lgb

    params = {#'num_leaves': 512,
         'num_leaves' : 32,
         'objective': 'multiclass',
         'max_depth': -1,
         'learning_rate': 0.01,
         "boosting": "gbdt",
         "feature_fraction": 0.9,
         "bagging_freq": 3,
         "bagging_fraction": 0.9,
         "bagging_seed": 11,
         "random_state": 42,          
         "verbosity": -1,
         "num_class": 5}

    # Additional parameters:
    early_stop = 500
    verbose_eval = 100
    num_rounds = 10000
    #n_splits = 5
    n_splits = 6
    
    from sklearn.model_selection import StratifiedKFold
    
    #kfold = StratifiedKFold(n_splits=n_splits)
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=15)

    #oof_train = np.zeros((X_train.shape[0]))
    #oof_test = np.zeros((X_test.shape[0], n_splits))

    i = 0
    val_qwks = []

    # Encode Label
    from sklearn.preprocessing import LabelEncoder

    label_encoder = LabelEncoder()
    #trainData.loc[trainData['Name'].isin(unnamedForms), 'Name'] = ""
    #X_train.loc[:, 'AdoptionSpeed'] = label_encoder.fit_transform(X_train['AdoptionSpeed'])
    #X_train['AdoptionSpeed'] = label_encoder.fit_transform(X_train['AdoptionSpeed'])
    X_train['AdoptionSpeed'] = label_encoder.fit_transform(X_train['AdoptionSpeed'])
    
    # Transform features into categorical.
    for c in categorical_features:
        #X_train[c] = X_train[c].astype('category')
        X_train[c] = X_train[c].astype('int')
        
    # Transform features into float.
    for c in numerical_features:
        X_train[c] = X_train[c].astype('float')
        
    # Transform features into bool.
    for c in bool_features:
        X_train[c] = X_train[c].astype('bool')
    
    for train_index, valid_index in kfold.split(X_train, X_train['AdoptionSpeed'].values):
    
        X_tr = X_train.iloc[train_index, :]
        X_val = X_train.iloc[valid_index, :]
    
        y_tr = X_tr['AdoptionSpeed'].values
        X_tr = X_tr.drop(['AdoptionSpeed'], axis=1)
    
        y_val = X_val['AdoptionSpeed'].values
        X_val = X_val.drop(['AdoptionSpeed'], axis=1)
    
        #print('\ny_tr distribution: {}'.format(Counter(y_tr)))
    
        d_train = lgb.Dataset(X_tr, label=y_tr)
        d_valid = lgb.Dataset(X_val, label=y_val)
        watchlist = [d_train, d_valid]
    
        print('training XGBoost:')
        # Predict using xgboost
        import xgboost as xgb
        xgb_model = xgb.XGBRegressor(booster = "gbtree", objective = "multi:softprob", num_class = 5, eval_metric = "mlogloss")
        xgb_model.fit(X_tr, y_tr)
        xgb_val_pred = xgb_model.predict(X_val)
        
        xgb_rounded_val_preds = []
        
        for pred in xgb_val_pred:
            xgb_rounded_val_preds.append(np.argmax(pred))
        
        print('training Random Forest:')
        # Predict using Random Forest
        from sklearn.ensemble import RandomForestClassifier
        randomForest = RandomForestClassifier(n_estimators=100, n_jobs=2, random_state=1357)
        randomForest.fit(X_tr, y_tr)
        
        randomForest_val_pred = randomForest.predict(X_val)
        
        randomForest_rounded_val_preds = []
        
        for pred in randomForest_val_pred:
            randomForest_rounded_val_preds.append(pred)
            
        print('training LGB')
        # Predict using LGB
        lgbModel = lgb.train(params,
                      train_set=d_train,
                      num_boost_round=num_rounds,
                      valid_sets=watchlist,
                      verbose_eval=verbose_eval,
                      early_stopping_rounds=early_stop)
        
        lgb_val_pred = lgbModel.predict(X_val, num_iteration=lgbModel.best_iteration)
        
        lgb_rounded_val_preds = []
        
        for pred in lgb_val_pred:
            lgb_rounded_val_preds.append(np.argmax(pred))
        
        
        
        from sklearn.tree import DecisionTreeClassifier
        print('training Decision Trees')
        decisionTree = DecisionTreeClassifier()
        decisionTree.fit(X_tr, y_tr)
        
        decisionTree_val_pred = decisionTree.predict(X_val)
        
        decisionTree_rounded_val_preds = []
        
        for pred in decisionTree_val_pred:
            decisionTree_rounded_val_preds.append(pred)
        
        
        from sklearn.naive_bayes import GaussianNB
        print('training Naive Bayes Classifier')
        naiveBayes = GaussianNB()
        naiveBayes.fit(X_tr, y_tr)
        
        naiveBayes_val_pred = naiveBayes.predict(X_val)
        
        naiveBayes_rounded_val_preds = []
        
        for pred in naiveBayes_val_pred:
            naiveBayes_rounded_val_preds.append(pred)
        
        
        #from sklearn import svm
        #print('training SVM')
        #linearSVM = svm.SVC(kernel='linear', C=1, decision_function_shape='ovo').fit(X_tr, y_tr)
        
        #linearSVM_val_pred = linearSVM.predict(X_val)
        
        #linearSVM_rounded_val_preds = []
        
        #for pred in linearSVM_val_pred:
        #    linearSVM_rounded_val_preds.append(pred)
        
        #test_pred = model.predict(X_test, num_iteration=model.best_iteration)
        #val_pred = model.predict(X_val)

        #xgb_d_train = xgb.DMatrix(data=X_tr, label=y_tr, feature_names=X_tr.columns)
        #xgb_d_valid = xgb.DMatrix(data=X_val, label=y_val, feature_names=X_val.columns)
        #model = xgb.train(dtrain=xgb_d_train, num_boost_round=num_rounds, evals=watchlist, early_stopping_rounds=early_stop, verbose_eval=verbose_eval, params=params)
        #model = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1, max_depth = 5, alpha = 10, n_estimators = 10)
        #model = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.85)
        #xgb_model = xgb.XGBRegressor(booster = "gbtree", objective = "multi:softprob", num_class = 5, eval_metric = "mlogloss")
        #booster = "gbtree", objective = "multi:softprob", num_class = 3, eval_metric = "mlogloss"
        #xgb_model.fit(X_tr, y_tr)
    
        #val_pred = model.predict(X_val, num_iteration=model.best_iteration)
        #test_pred = model.predict(X_test, num_iteration=model.best_iteration)
        #val_pred = model.predict(X_val)
    
        #oof_train[valid_index] = val_pred
        #oof_test[:, i] = test_pred
        
        # Merging all predictions.
        val_pred = []
        
        for pred in xgb_rounded_val_preds:
            val_pred.append([pred])
            
        for i in range(len(val_pred)):
            val_pred[i].append(randomForest_rounded_val_preds[i])
            
        for i in range(len(val_pred)):
            val_pred[i].append(lgb_rounded_val_preds[i])
            
        for i in range(len(val_pred)):
            val_pred[i].append(decisionTree_rounded_val_preds[i])
            
        for i in range(len(val_pred)):
            val_pred[i].append(naiveBayes_rounded_val_preds[i])
            
            
        #for i in range(len(val_pred)):
        #    val_pred[i].append(linearSVM_rounded_val_preds[i])
        
        # Computing  QWK
        from sklearn.metrics import cohen_kappa_score, confusion_matrix

        #lgb_train_preds = lgb_model.predict(X_train_lgb, num_iteration=lgb_model.best_iteration)
        #train_actuals = X_train_lgb['AdoptionSpeed'].values

        #print("y_val: " + str(y_val[:5]))
        #print("val_pred: " + str(val_pred[:5]))
        rounded_val_preds = []
        
        #for pred in val_pred:
        #    minDiff = pred
        #    minClass = 0
        #    for c in range(5):
        #        if abs(c - pred) < minDiff:
        #            minDiff = abs(c - pred)
        #            minClass = c
        #    rounded_val_preds.append(minClass)
        
        for pred in val_pred:
            rounded_val_preds.append(np.bincount(pred).argmax())
        
        print("rounded_val_pred: " + str(rounded_val_preds[:5]))
        qwk = cohen_kappa_score(y_val, rounded_val_preds, weights="quadratic")
        val_qwks.append(qwk)
        print("QWK score: " + str(qwk))
    
        i += 1
    print("Average Validation QWK: " + str(np.mean(val_qwks)))
    
    # Return the latest trained k-fold, perhaps we want to train on the entire data or return the one with the best validation score.
    return [xgb_model, randomForest, decisionTree, naiveBayes, lgbModel]
models = train_models(X_train_lgb, X_test_lgb, lgb_categorical_features, lgb_numerical_features, lgb_bool_features)
xgb_model = models[0]
randomForest = models[1]
decisionTree = models[2]
naiveBayes = models[3]
lgbModel = models[4]
# Save the models as a Pickle files.
import joblib

#os.makedirs("outputs", exist_ok=True)
#joblib.dump(value=xgb_model, filename="outputs/xgb.pkl")

#os.makedirs("outputs", exist_ok=True)
#joblib.dump(value=randomForest, filename="outputs/randomForest.pkl")

#os.makedirs("outputs", exist_ok=True)
#joblib.dump(value=decisionTree, filename="outputs/decisionTree.pkl")

#os.makedirs("outputs", exist_ok=True)
#joblib.dump(value=naiveBayes, filename="outputs/naiveBayes.pkl")

#os.makedirs("outputs", exist_ok=True)
#joblib.dump(value=lgbModel, filename="outputs/lgbModel.pkl")
import lightgbm

print(joblib.__version__)
print(lightgbm.__version__)

import sys
print(sys.version)

import pandas
print(pandas.__version__)

import numpy
print(numpy.__version__)

import xgboost
print(xgboost.__version__)
# Test predictions on fresh data
testData = pd.read_csv('/kaggle/input/petfinder-adoption-prediction/test/test.csv')
#lgb_test_pred = lgbModel.predict(X_val, num_iteration=lgbModel.best_iteration)

# Add Sentiment feature
# Sentiment Analysis
# Create dataframe for Sentiment Analysis

petIds = []
magnitudes = []
scores = []

for sentimentFilename in os.listdir("/kaggle/input/petfinder-adoption-prediction/test_sentiment"):
    with open("/kaggle/input/petfinder-adoption-prediction/test_sentiment/" + sentimentFilename, 'r') as f:
        jsonContent = json.loads(f.read())
        magnitude = jsonContent['documentSentiment']['magnitude']
        score = jsonContent['documentSentiment']['score']
        petIds.append(sentimentFilename.split('.')[0])
        magnitudes.append(float(magnitude))
        scores.append(float(score))
        
rows = []
for i in range(len(petIds)):
    rows.append([petIds[i], magnitudes[i], scores[i]])
columns = ['PetID', 'SentimentMagnitude', 'SentimentScore']

testSentimentDf=pd.DataFrame(rows)
testSentimentDf.columns = columns

testSentimentDf.head(10)
testDataWithSentiment = pd.merge(testData, testSentimentDf, on='PetID')
testDataWithSentiment.head(100)
def prepareDataset(df):
    
    # Clean up the 'Name' column.
    df.loc[df['Name'].isna(), 'Name'] = ""
    
    unnamedForms = set(['Lost Dog', 'No Name', '9 Puppies For Adoption!', 'No Name Yet', 'Not Named', '(No Name)', '[No Name]', '$ To Be Named $', 'Noname', 'Unamed Yet 2',\
               'Unamed', 'Unnamed', 'No Names Yet', 'Not Named Yet', 'Unnamed 3 Kittens ( By Dani)', 'No Name Kitten', 'Nameless', '(no Name)', 'Name Them & Love Them', \
               'Not Name Yet', 'No Names Yet', '*No Name*', '"no Name"', '(No Names Yet)', '* To Be Named *', 'Unnamed.', 'NO NAME', 'Not Yet Name', 'No Name Kitties', \
               'Waiting For You To Give Him A Name', 'No Names Yet', '*please Name Us*', 'Newborn *no Name', '- To Be Named -', 'No Name Yet, It\'s Up To The Owner', \
               'Name Them & Love Them 3', 'NO NAME YET', '(No Name - She Is Just A Stray)', 'Cream Cat (unnamed)', '(no Name)', 'Wait For The Real Owner To Name It', \
               '4 Kittens Open For Adoption (no Name)', 'Need You Giving  A Name', 'No Name 2', 'UNNAMED', 'Unamed Yet', 'No Name Yet....', 'Kitten....no Name', \
               'Name Less Kitten', 'Haven\'t Named Them', 'No Name Yet (Must Neuter)', 'Haven\'t Name Yet', 'Haven\'t Been Named', 'Not Yet Named'])

    df.loc[trainData['Name'].isin(unnamedForms), 'Name'] = ""
    
    codeNames = set()
    for name in df['Name']:
        strName = str(name).lower()
        if len(strName) < 3 or ('a' not in strName and 'e' not in strName and 'i' not in strName and 'o' not in strName and 'u' not in strName and 'y' not in strName):
            codeNames.add(strName)
    
    df.loc[df['Name'].isin(codeNames), 'Name'] = ""
    
    
    # Create new features.
    categorical_features = ['Type', 'Breed1', 'Breed2', 'Gender', 'Color1', 'Color2', 'Color3', 'Health', 'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed', \
                           'Sterilized']
    
    bool_features = ['HasPhoto', 'IsSinglePet', 'IsFree', 'IsPureBreed', 'HasVideo']

    numerical_features = ['Age', 'PhotoAmt', 'Age_Years', 'Fee', 'SentimentScore', 'SentimentMagnitude', 'Name_Length', 'SentimentMultiplier', \
                          'DescriptionLength', 'Quantity', 'VideoAmt']
    
    df['Name_Length'] = df['Name'].map(str).apply(len)
    df['Age_Years'] = df['Age'] // 12
    df['IsPureBreed'] = (df['Breed1'] == 0) | (df['Breed2'] == 0) | (df['Breed1'] == df['Breed2'])
    df['HasPhoto'] = df['PhotoAmt'] > 0
    df['SentimentMultiplier'] = df['SentimentScore'] * df['SentimentMagnitude']
    df['DescriptionLength'] = df['Description'].map(str).apply(len)
    df['IsSinglePet'] = df['Quantity'] == 1
    df['IsFree'] = df['Fee'] == 0
    df['HasVideo'] = df['VideoAmt'] > 0
    
    # Transform features into categorical.
    for c in categorical_features:
        df[c] = df[c].astype('int')
        
    # Transform features into float.
    for c in numerical_features:
        df[c] = df[c].astype('float')
        
    # Transform features into bool.
    for c in bool_features:
        df[c] = df[c].astype('bool')
        
    #Filter out columns.
    allFeatures = ['Type', 'Age', 'Breed1', 'Breed2', 'Gender', 'Color1', 'Color2', \
       'Color3', 'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed', \
       'Sterilized', 'Health', 'Fee', 'SentimentScore', 'SentimentMagnitude', \
       'Name_Length', 'Age_Years', 'IsPureBreed', 'HasPhoto', 'SentimentMultiplier', \
       'PhotoAmt', 'DescriptionLength', 'Quantity', 'IsSinglePet', 'IsFree', 'VideoAmt', 'HasVideo']
    
    #Filter out columns
    return df[allFeatures]
testDataWithSentiment = prepareDataset(testDataWithSentiment)

testDataWithSentiment.head(10)
lgb_test_pred = lgbModel.predict(testDataWithSentiment, num_iteration=lgbModel.best_iteration)
# Get feature importance for Random Forest
if 'AdoptionSpeed' in X_train_lgb.columns:
    X_train_lgb = X_train_lgb.drop(['AdoptionSpeed'], axis=1)

print("Train columns: " + str(X_train_lgb.columns))
print(randomForest.feature_importances_)

randomForestFeatureImportances = []
for i in range(len(randomForest.feature_importances_)):
    randomForestFeatureImportances.append(tuple([X_train_lgb.columns[i], randomForest.feature_importances_[i]]))

print("Random Forest feature importances")
print(sorted(randomForestFeatureImportances, key=lambda x: x[1], reverse=True))

#xgb_model = models[0]
#randomForest = models[1]
#decisionTree = models[2]
#naiveBayes = models[3]
#lgbModel = models[4]
# Get feature importance for Decision Trees
if 'AdoptionSpeed' in X_train_lgb.columns:
    X_train_lgb = X_train_lgb.drop(['AdoptionSpeed'], axis=1)

print("Train columns: " + str(X_train_lgb.columns))
print(decisionTree.feature_importances_)

decisionTreeFeatureImportances = []
for i in range(len(randomForest.feature_importances_)):
    decisionTreeFeatureImportances.append(tuple([X_train_lgb.columns[i], decisionTree.feature_importances_[i]]))

print("Decision Tree feature importances")
print(sorted(decisionTreeFeatureImportances, key=lambda x: x[1], reverse=True))
# Get feature importance for XGB
if 'AdoptionSpeed' in X_train_lgb.columns:
    X_train_lgb = X_train_lgb.drop(['AdoptionSpeed'], axis=1)

print("Train columns: " + str(X_train_lgb.columns))
print(xgb_model.feature_importances_)

xgbModelFeatureImportances = []
for i in range(len(xgb_model.feature_importances_)):
    xgbModelFeatureImportances.append(tuple([X_train_lgb.columns[i], xgb_model.feature_importances_[i]]))

print("XGB feature importances")
print(sorted(xgbModelFeatureImportances, key=lambda x: x[1], reverse=True))
