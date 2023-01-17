#Importing libraries



import os

import pandas as pd 

import numpy as np



from IPython.display import display

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

sns.set(style='white', context='notebook', palette='deep')



colors = ["#66c2ff", "#5cd6d6", "#00cc99", "#85e085", "#ffd966", "#ffb366", "#ffb3b3", "#dab3ff", "#c2c2d6"]

sns.set_palette(palette = colors, n_colors = 4)



print('Data Manipulation, Mathematical Computation and Visualisation packages imported!')



import warnings

def ignore_warn(*args, **kwargs):

    pass

warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)

warnings.filterwarnings("ignore", category=DeprecationWarning)

print('Deprecation warning will be ignored!')
#Uploading the 5 excel sheets consisting the GTD data



xl_1=pd.ExcelFile(r"../input/gtd1993_0718dist.xlsx") #Uploading Excel sheet1

xl_2=pd.ExcelFile(r"../input/gtd_14to17_0718dist.xlsx") #Uploading Excel sheet2

xl_3=pd.ExcelFile(r"../input/gtd_70to95_0718dist.xlsx") #Uploading Excel sheet3

xl_4=pd.ExcelFile(r"../input/gtd_96to13_0718dist.xlsx") #Uploading Excel sheet4

#xl_5=pd.ExcelFile(r"C:\Users\aniru\OneDrive\Desktop\Studies\Data Science\Fellowship.AI\data5.xlsx") #Uploading Excel sheet5



print('Excel Files of GTD Uploaded!')
#Parsing the excel sheets to be suitable for pandas dataframe. 

#The sheets are named as "Data" in all the five GTD excel files.



df_1= xl_1.parse("Data") #Excel sheet1 parsed into dataframe1 (df_1)

df_2= xl_2.parse("Data") #Excel sheet2 parsed into dataframe2 (df_2)

df_3= xl_3.parse("Data") #Excel sheet3 parsed into dataframe3 (df_3)

df_4= xl_4.parse("Data") #Excel sheet4 parsed into dataframe4 (df_4)

#df_5= xl_5.parse("Data") #Excel sheet5 parsed into dataframe5 (df_5)



print('Excel sheets converted to DataFrames!')
#Appending all the 5 dataframes into a single dataframe



#df_2 has been appended into df_1

df_1=df_1.append(df_2, ignore_index=True)

#df_3 has been appended into df_1

df_1=df_1.append(df_3, ignore_index=True)

#df_4 has been appended into df_1

df_1=df_1.append(df_4, ignore_index=True)



# The number of rows and columns of terror_df, found out by pd.Dataframe.shape



#df_1 acts as a backup dataframe 

terror_df=df_1.copy()



terror_df.head(10)



#terror_df is the copy of df1 where all other dataframes have been appended
terror_df.shape
# To avoid confusion,the dataset is restricted to only attacks that were of terrorist nature.



terror_df = terror_df[(terror_df.crit1 == 1) & (terror_df.crit2 == 1) & (terror_df.crit3 == 1) & (terror_df.doubtterr == 0)]
#missing value counts in each of these columns



miss = terror_df.isnull().sum()/len(terror_df)

miss = miss[miss > 0]

miss.sort_values(inplace=True)

miss

# Removing columns having more than 20% of missing values



frac = len(terror_df) * 0.8

terror_df=terror_df.dropna(thresh=frac, axis=1)

terror_df= terror_df.replace(-9,0)

terror_df= terror_df.drop(["city"], axis=1)

print('Removing columns having 20% of missing values')
#missing value counts in each of these columns after removing columns having more than 20% of missing values



miss = terror_df.isnull().sum()/len(terror_df)

miss = miss[miss > 0]

miss.sort_values(inplace=True)

miss
#replacing few strings with generalised strings to be common for all rows



terror_df.weaptype1_txt.replace(

    'Vehicle (not to include vehicle-borne explosives, i.e., car or truck bombs)',

    'Vehicle', inplace = True)



terror_df.gname = terror_df.gname.str.lower()
# replacing 'nkill', 'nwound', 'ncasualties' with has_casualities boolean



terror_df['ncasualties'] = terror_df['nkill'] + terror_df['nwound']

terror_df['has_casualties'] = terror_df['ncasualties'].apply(lambda x: 0 if x == 0 else 1)

terror_df= terror_df.drop(['nkill', 'nwound', 'ncasualties'], axis=1)
#visualising missing values



miss = terror_df.isnull().sum()/len(terror_df)

miss = miss[miss > 0]

miss = miss.to_frame()

miss.columns = ['count']

miss.index.names = ['Name']

miss['Name'] = miss.index



#plot the missing value count

sns.set(style="whitegrid", color_codes=True)

sns.barplot(x = 'Name', y = 'count', data=miss)

plt.xticks(rotation = 90)

plt.show()
# imputing with mode

terror_df['specificity'] = terror_df['specificity'].fillna(terror_df['specificity'].mode()[0])

terror_df['natlty1'] = terror_df['natlty1'].fillna(terror_df['natlty1'].mode()[0])



#imputing with value '0'

terror_df['guncertain1'] = terror_df['guncertain1'].fillna(terror_df['guncertain1'].fillna(0))

terror_df['targsubtype1'] = terror_df['targsubtype1'].fillna(terror_df['targsubtype1'].fillna(0))

terror_df['weapsubtype1'] = terror_df['weapsubtype1'].fillna(terror_df['weapsubtype1'].fillna(0))

terror_df['ishostkid'] = terror_df['ishostkid'].fillna(terror_df['ishostkid'].fillna(0))

terror_df['latitude'] = terror_df['latitude'].fillna(terror_df['latitude'].fillna(0))

terror_df['longitude'] = terror_df['longitude'].fillna(terror_df['longitude'].fillna(0))



print("Numerical variables Imputed!")
for col in ("target1","natlty1_txt", "provstate", "targsubtype1_txt", "weapsubtype1_txt"):

    terror_df[col] = terror_df[col].fillna("None")

print("Categorical variables Imputed!")

unknown_df= terror_df[terror_df['gname'] == "unknown"]
# removing all unknown_df values from terror_df



terror_df=terror_df[~terror_df.isin(unknown_df)]

terror_df= terror_df.dropna()
# dropping "iyear", "eventid", "imonth", "iday", "extended" as they are redundant.



terror_df=terror_df.drop(["iyear", "eventid", "imonth", "iday", "extended"], axis=1)

terror_df= terror_df.dropna()



unknown_df=unknown_df.drop(["iyear", "eventid", "imonth", "iday", "extended"], axis=1)

unknown_df= unknown_df.dropna()
# All the gname stored into an another dataset Y



Y = pd.DataFrame()

Y = terror_df["gname"]

Y_count= Y.value_counts()
print(Y_count)
# Binning whole gname values based on their frequency as explained in above steps.



Y_bins=pd.qcut(Y_count, [0, .5, .75, 1], labels=["Small Org", "Medium Org" ,"Dangerous Groups"])

Y_bins= Y_bins.to_dict()



Y=Y.reset_index()

Y=Y.drop(["index"], axis=1)



Label = []



for x in Y["gname"]:

    for key, value in Y_bins.items():

           if x==key:

            Label.append(value)



Y["Label"]= pd.Series(Label)
# Example of dataset after Binning 



Y['Label'].head(5)
# Verifying the Binning process with barchart



plt.subplot(1, 3, 3)

sns.barplot(x="Label", y=Y.index, data=Y);
# Label encoding the "Small Org", "Medium Org", and "Dangerous Groups" into integers as they are ordinal variables and as it will be easier for training the model



Y['Label'] = Y['Label'].map({"Small Org":1, "Medium Org":2, "Dangerous Groups":3})

Y['Label'].shape
#Storing the binned and encoded terror organisation names back to terror_df dataset.



terror_df["Label"]= Y["Label"].values
# As gname has been binned and encoded into "Label" column, "gname" is dropped. "Label" now acts as a label dataset while training.



terror_df= terror_df.drop(["gname"], axis=1)
#preparing the correlation matrix



corr = terror_df.corr()

plt.subplots(figsize=(30, 30))

cmap = sns.diverging_palette(150, 250, as_cmap=True)

sns.heatmap(corr, cmap="RdYlBu", vmax=1, vmin=-0.6, center=0.2, square=True, linewidths=0, cbar_kws={"shrink": .5}, annot = True);
print (corr['Label'].sort_values(ascending=False)[:15], '\n') #top 15 values

print ('----------------------')

print (corr['Label'].sort_values(ascending=False)[-5:]) #last 5 values`
#Dropping the gname column from the unknown_df dataset. Its filled with "unknown" values thus its no use for us.



unknown_df= unknown_df.drop(["gname"], axis= 1)
# Storing Label column from terror_df as our Label Dataset for training process



y_train = terror_df.Label.values
# Dropping Label from terror_df as we will later convert this dataset into train dataset.



terror_df= terror_df.drop(["Label"], axis= 1)
# Concating terror_df and unknown_df into one column for encoding categorical variables and name it as all_data



# Storing shape of train dataset and test dataset for later use

ntrain = terror_df.shape[0] # terror_df is train dataset

ntest = unknown_df.shape[0] # unknown_df is test dataset



all_data= pd.concat((terror_df, unknown_df)).reset_index(drop=True)



print("all_data shape: {}".format(all_data.shape))
cols = terror_df.columns



# List all columns having categorical variables



num_cols = terror_df._get_numeric_data().columns

list(set(cols) - set(num_cols))
# One-hot encoding all nominal variables



all_data["targtype1_txt"] = pd.get_dummies(all_data["targtype1_txt"])

all_data["region_txt"] = pd.get_dummies(all_data["region_txt"])

all_data["country_txt"] = pd.get_dummies(all_data["country_txt"])

all_data["dbsource"] = pd.get_dummies(all_data["dbsource"])

all_data["provstate"] = pd.get_dummies(all_data["provstate"])
#Label encoding all categorical variables



from sklearn import preprocessing



le = preprocessing.LabelEncoder()

le.fit(all_data["attacktype1_txt"])

list(le.classes_)

all_data["attacktype1_txt"]= pd.Series(list(le.transform(all_data["attacktype1_txt"]))) 



le_2 = preprocessing.LabelEncoder()

le_2.fit(all_data["weapsubtype1_txt"])

list(le_2.classes_)

all_data["weapsubtype1_txt"]= pd.Series(list(le_2.transform(all_data["weapsubtype1_txt"]))) 



le_3 = preprocessing.LabelEncoder()

le_3.fit(all_data["target1"])

list(le_3.classes_)

all_data["target1"]= pd.Series(list(le_3.transform(all_data["target1"]))) 



le_4 = preprocessing.LabelEncoder()

le_4.fit(all_data["weaptype1_txt"])

list(le_4.classes_)

all_data["weaptype1_txt"]= pd.Series(list(le_4.transform(all_data["weaptype1_txt"]))) 



le_5 = preprocessing.LabelEncoder()

le_5.fit(all_data["targsubtype1_txt"])

list(le_5.classes_)

all_data["targsubtype1_txt"]= pd.Series(list(le_5.transform(all_data["targsubtype1_txt"]))) 



le_6 = preprocessing.LabelEncoder()

le_6.fit(all_data["natlty1_txt"])

list(le_6.classes_)

all_data["natlty1_txt"]= pd.Series(list(le_6.transform(all_data["natlty1_txt"]))) 
# Scaling all the values so that they are in the same scale



from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

all_data=sc.fit_transform(all_data)

# As scaling turns all_data to numpy, we convert it back to pandas for later use.



all_data = pd.DataFrame(all_data)
#creating train and test datasets



train = all_data[:ntrain]

test = all_data[ntrain:]



print(train.shape)

print(test.shape)
from xgboost import XGBClassifier

from xgboost import plot_importance

model = XGBClassifier()

model.fit(train, y_train)
# Sort feature importances from GBC model trained earlier

indices = np.argsort(model.feature_importances_)[::-1]

indices = indices[:75]



# Visualise these with a barplot

plt.subplots(figsize=(20, 15))

g = sns.barplot(y=cols[indices], x = model.feature_importances_[indices], orient='h', palette = colors)

g.set_xlabel("Relative importance",fontsize=12)

g.set_ylabel("Features",fontsize=12)

g.tick_params(labelsize=9)

g.set_title("XGB feature importance");
# Storing and copying train and test values for safety and maybe later use



train_2 = train.copy()

test_2 = test.copy()
from sklearn.feature_selection import SelectFromModel



# Allow the feature importances attribute to select the most important features

xgb_feat_red = SelectFromModel(model, prefit = True)



# Reduce estimation, validation and test datasets

train = xgb_feat_red.transform(train)

test = xgb_feat_red.transform(test)





print("Results of 'feature_importances_':")

print('X_train: ', train.shape, '\nX_test: ', test.shape)
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score



# Dividing train dataset and y_train dataset to X_train, X_test, Y_train, Y_test respectively for training the model.



X_train, X_test, Y_train, Y_test = train_test_split(train, y_train, test_size=0.3, random_state=42, stratify=y_train)
print('X_train: ', X_train.shape, '\nX_test: ', X_test.shape, '\nY_train: ', Y_train.shape, '\nY_test: ', Y_test.shape)
# Importing Statistical packages!

# Importing Metrics packages!



from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score, train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import ShuffleSplit

from sklearn.metrics import confusion_matrix

from sklearn.metrics import make_scorer

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler



print('Validation and metric libraries imported')
# Importing Algorithm packages!



from sklearn.neural_network import MLPClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB



print('Classification Models imported')
names = ["K-Nearest Neighbors", "Linear SVM",  "Naive Bayes", "Decision Tree", "Neural Net"]



models= [ KNeighborsClassifier(3), SVC(kernel="linear", C=0.025), GaussianNB(), DecisionTreeClassifier(max_depth=5), MLPClassifier(alpha=1)]
# First I will use ShuffleSplit as a way of randomising the cross validation samples.

shuff = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)



#create table to compare MLA metrics

columns = ['Name', 'Parameters', 'Train Accuracy Mean', 'Test Accuracy']

before_model_compare = pd.DataFrame(columns = columns)



#index through models and save performance to table

row_index = 0

for alg in models:



    #set name and parameters

    model_name = alg.__class__.__name__

    before_model_compare.loc[row_index, 'Name'] = model_name

    before_model_compare.loc[row_index, 'Parameters'] = str(alg.get_params())

    

    alg.fit(X_train, Y_train)

    

    #score model with cross validation: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate

    training_results = cross_val_score(alg, X_train, Y_train, cv = shuff, scoring= 'accuracy').mean()

    test_results = accuracy_score(Y_test, alg.predict(X_test)).mean()

    

    before_model_compare.loc[row_index, 'Train Accuracy Mean'] = (training_results)*100

    before_model_compare.loc[row_index, 'Test Accuracy'] = (test_results)*100

    

    row_index+=1

    print(row_index, alg.__class__.__name__, 'trained...')



decimals = 3

before_model_compare['Train Accuracy Mean'] = before_model_compare['Train Accuracy Mean'].apply(lambda x: round(x, decimals))

before_model_compare['Test Accuracy'] = before_model_compare['Test Accuracy'].apply(lambda x: round(x, decimals))

before_model_compare
models= [ KNeighborsClassifier(), SVC(), GaussianNB(), DecisionTreeClassifier(), MLPClassifier()]



KNN_param_grid = {'n_neighbors': [1,2,3]}

SVC_param_grid = {'C': [0.025, 0.030, 0.020]}

G_param_grid = {'priors': [None, None]}

DTC_param_grid = {'max_depth': range(5, 30, 2)}

MLP_param_grid = {'alpha': [1, 2, 3]}



params_grid = [KNN_param_grid, SVC_param_grid, G_param_grid, DTC_param_grid, MLP_param_grid]

#create table to compare MLA metrics

columns = ['Name', 'Parameters', 'Train Accuracy Mean', 'Test Accuracy']

after_model_compare = pd.DataFrame(columns = columns)



row_index = 0

for alg in models:

    

    gs_alg = GridSearchCV(alg, param_grid = params_grid[0], cv = shuff, scoring= 'accuracy', n_jobs=-1)

    

    params_grid.pop(0)

    #set name and parameters

    model_name = alg.__class__.__name__

    after_model_compare.loc[row_index, 'Name'] = model_name

    

    gs_alg.fit(X_train, Y_train)

    gs_best = gs_alg.best_estimator_

    after_model_compare.loc[row_index, 'Parameters'] = str(gs_alg.best_params_)

    

    #score model with cross validation: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate

    

    after_training_results = gs_alg.best_score_

    after_test_results = accuracy_score(Y_test, gs_alg.predict(X_test)).mean()

    

    after_model_compare.loc[row_index, 'Train Accuracy Mean'] = (after_training_results)*100

    after_model_compare.loc[row_index, 'Test Accuracy'] = (after_test_results)*100

    

    row_index+=1

    print(row_index, alg.__class__.__name__, 'trained...')



decimals = 3

after_model_compare['Train Accuracy Mean'] = after_model_compare['Train Accuracy Mean'].apply(lambda x: round(x, decimals))

after_model_compare['Test Accuracy'] = after_model_compare['Test Accuracy'].apply(lambda x: round(x, decimals))

after_model_compare



models= [ KNeighborsClassifier(), SVC(), GaussianNB(), DecisionTreeClassifier(), MLPClassifier()]



KNN_param_grid = {'n_neighbors': [1,2,3]}

SVC_param_grid = {'C': [0.025, 0.030, 0.020]}

G_param_grid = {'priors': [None, None]}

DTC_param_grid = {'max_depth': range(5, 30, 2)}

MLP_param_grid = {'alpha': [1, 2, 3]}



params_grid = [KNN_param_grid, SVC_param_grid, G_param_grid, DTC_param_grid, MLP_param_grid]



stacked_validation_train = pd.DataFrame()

stacked_test_train = pd.DataFrame()



row_index=0



for alg in models:

    

    gs_alg = GridSearchCV(alg, param_grid = params_grid[0], cv = shuff, scoring = 'accuracy', n_jobs=-1)

    params_grid.pop(0)

    

    gs_alg.fit(X_train, Y_train)

    gs_best = gs_alg.best_estimator_

    stacked_validation_train.insert(loc = row_index, column = names[0], value = gs_best.predict(X_test))

    print(row_index+1, alg.__class__.__name__, 'predictions added to stacking validation dataset...')

    

    stacked_test_train.insert(loc = row_index, column = names[0], value = gs_best.predict(test))

    print(row_index+1, alg.__class__.__name__, 'predictions added to stacking test dataset...')

    print("-"*50)

    names.pop(0)

    

    row_index+=1

    

print('Done')
stacked_validation_train.head(5)
stacked_test_train.head(5)
# First drop the Lasso results from the table, as we will be using Lasso as the meta-model

drop = ['K-Nearest Neighbors']

stacked_validation_train.drop(drop, axis=1, inplace=True)

stacked_test_train.drop(drop, axis=1, inplace=True)



# Now fit the meta model and generate predictions

meta_model = make_pipeline(RobustScaler(),KNeighborsClassifier(1))

meta_model.fit(stacked_validation_train, Y_test)



meta_model_pred = meta_model.predict(stacked_test_train)

print("Meta-model trained and applied!...")
models= [ KNeighborsClassifier(), SVC(), GaussianNB(), DecisionTreeClassifier(), MLPClassifier()]

names = ["K-Nearest Neighbors", "Linear SVM",  "Naive Bayes", "Decision Tree", "Neural Net"]

KNN_param_grid = {'n_neighbors': [1,2,3]}

SVC_param_grid = {'C': [0.025, 0.030, 0.020]}

G_param_grid = {'priors': [None, None]}

DTC_param_grid = {'max_depth': range(5, 30, 2)}

MLP_param_grid = {'alpha': [1, 2, 3]}



params_grid = [KNN_param_grid, SVC_param_grid, G_param_grid, DTC_param_grid, MLP_param_grid]





final_predictions = pd.DataFrame()



row_index=0



for alg in models:

    

    gs_alg = GridSearchCV(alg, param_grid = params_grid[0], cv = shuff, scoring = 'accuracy', n_jobs=-1)

    params_grid.pop(0)

    

    gs_alg.fit(stacked_validation_train, Y_test)

    gs_best = gs_alg.best_estimator_

    final_predictions.insert(loc = row_index, column = names[0], value = gs_best.predict(stacked_test_train))

    print(row_index+1, alg.__class__.__name__, 'final results predicted added to table...')

    names.pop(0)

    

    row_index+=1



print("-"*50)

print("Done")

    

final_predictions.head()
ensemble = meta_model_pred*(2.5/10)+ final_predictions["K-Nearest Neighbors"]*(1.5/10)+final_predictions["Linear SVM"]*(1.5/10)+ final_predictions["Naive Bayes"]*(1.5/10)+ final_predictions["Decision Tree"]*(1.5/10)+ final_predictions["Neural Net"]*(1.5/10)



#Final predicted values for the attacks that have been not attrbuted any attacks.



unknown_df["gname"]= ensemble.values



unknown_df.head(5)