#@author: Abhishek Kumar Gauraw
# 1.1 Data manipulation modules
import pandas as pd        # R-like data manipulation
import numpy as np         # n-dimensional arrays

# 1.2 For pltting
import matplotlib.pyplot as plt      # For base plotting
# Seaborn is a library for making statistical graphics
# in Python. It is built on top of matplotlib and 
#  numpy and pandas data structures.
import seaborn as sns                # Easier plotting

# 1.3 Misc
import os


############## Data Loading and Exploration ##################
# 2.1 Read data file
data_test = pd.read_csv("../input/test.csv")
data_train = pd.read_csv("../input/train.csv")


# 2.2 Explore data
print("Dimension of Test DataSet :" ) 
data_test.shape                         # dim()

print("Dimension of Training DataSet :")
data_train.shape                        # dim()

print("Columns of Test DataSet :")              
data_test.columns

print("Columns of Training DataSet :")
data_train.columns

print("Gllimpse/Summary of Test DataSet :")
data_test.describe()                      # summary()

print("Gllimpse/Summmary of Training DataSet :")
data_train.describe()                     # summary()

print("First five records of Test DataSet :")
data_test.head()                          # Top 5 records of test data

print("First five records of Training DataSet :")
data_train.head()                          # Top 5 records of training data

# Identifying and Removing columns not useful in analysis
missing_col_train = data_train.isnull().sum().sort_values(0, ascending = False)
print("Missing or Null Value column: ")
missing_col_train.head()
# Looking at this columns rez_esc,v18q1 and v2al have many null/missing values
## Hence we can remove these 3 columns to avoid any issue in modelling further

data_train.drop(["rez_esc",
              "v18q1",
              "v2a1",
    ], axis=1, inplace=True)

del(missing_col_train)

print("Removed 3 columns")

## Remove these 3 columns from test data to avoid any issue in modelling further

data_test.drop(["rez_esc",
              "v18q1",
              "v2a1",
    ], axis=1, inplace=True)

print("Removed 3 columns from Test data")
print("New Dimension of Training DataSet :")
data_train.shape  
#We can finish off with the meaneduc and SQBmeaned label by imputing them with the median of the columns.
median_meaneduc = data_train['meaneduc'].median()
median_SQBmeaned = data_train['SQBmeaned'].median()
data_train['meaneduc'] = data_train['meaneduc'].fillna(median_meaneduc)
data_train['SQBmeaned'] = data_train['SQBmeaned'].fillna(median_SQBmeaned)

median_meaneduc_test = data_test['meaneduc'].median()
median_SQBmeaned_test = data_test['SQBmeaned'].median()
data_test['meaneduc'] = data_test['meaneduc'].fillna(median_meaneduc_test)
data_test['SQBmeaned'] = data_test['SQBmeaned'].fillna(median_SQBmeaned_test)
data_train.loc[data_train['Target'].isin([1]),'target_des'] = "Extereme Poverty"         
data_train.loc[data_train['Target'].isin([2]),'target_des'] = "Vulnerable"         
data_train.loc[data_train['Target'].isin([3]),'target_des'] = "Moderate Poverty"         
data_train.loc[data_train['Target'].isin([4]),'target_des'] = "NonVulnerable"         

data_train['Target'].value_counts()
#Target - the target is an ordinal variable indicating groups of income levels. 
## 1 = extreme poverty 
## 2 = moderate poverty 
## 3 = vulnerable households 
## 4 = non vulnerable households

# Count Plot for Group of different Income levels
income_lvl_plot = sns.countplot("target_des", data = data_train)
income_lvl_plot.set_title("Group of Income Levels at Cost Rica")
income_lvl_plot.set_xticklabels(income_lvl_plot.get_xticklabels(), rotation=45)

# Violin Plot to determine gender wise distribution along with poverty level 
genderwise_total = data_train[["r4h3", "r4m3"]].groupby(data_train["target_des"]).sum()
print(genderwise_total)
gender_plot = (sns.violinplot(data=genderwise_total,
               split=True,         # If hue variable has two levels, draw half of a violin for each level.
               inner="quartile"    #  Options: “box”, “quartile”, “point”, “stick”, None 
               )
        .set_xticklabels(['Male','Female'])    
)

male_plot = (sns.violinplot( y=data_train["target_des"], x=data_train["r4h3"] )
           .set(xlabel='Male', ylabel='Poverty level')  
           )



female_plot = (sns.violinplot( y=data_train["target_des"], x=data_train["r4m3"] )
            .set(xlabel='Female', ylabel='Poverty level')
            )

del(genderwise_total)

# Education level of people from Costa-Rica
Edu_level_total = data_train[["instlevel1", "instlevel2", "instlevel3","instlevel4", "instlevel5", "instlevel6","instlevel7", "instlevel8", "instlevel9"]].groupby(data_train["target_des"]).sum()

print(Edu_level_total)
labels = ['No level of education', 'Incomplete primary', 'Complete primary', 'Incomplete academic secondary level','Complete academic secondary level','Incomplete technical secondary level','Complete technical secondary level','Undergraduate and higher education','Postgraduate higher education']
sns.set_style("whitegrid")

edu_lvl_plot = (
   sns.violinplot(data=Edu_level_total,
               split=True,         
               inner="quartile"     
               )
   .set(xlabel='Education Level', ylabel='Poverty level')
        
)

edu_lvl_plot = (
   sns.violinplot(data=Edu_level_total,
               split=True,         
               inner="quartile"     
               )
    .set_xticklabels(labels,rotation=20)
)

del(Edu_level_total)
#Check for households where The household population has unequal target distribution
all_equal = data_train.groupby('idhogar')['Target'].apply(lambda x: x.nunique() == 1)
not_equal = all_equal[all_equal != True]
print(len(not_equal))
#correcting the unqual households
for household in not_equal.index:
    true_target = int(data_train[(data_train['idhogar'] == household) & (data_train['parentesco1'] == 1.0)]['Target'])
    data_train.loc[data_train['idhogar'] == household, 'Target'] = true_target
data_train.fillna(-1, inplace = True)
data_test.fillna(-1, inplace = True)
data_train['dependency'] = np.sqrt(data_train['SQBdependency'])
data_test['dependency'] = np.sqrt(data_test['SQBdependency'])
def mapping(data):
    if data == 'yes':
        return 1
    elif data == 'no':
        return 0
    else:
        return data
data_train['dependency'] = data_train['dependency'].apply(mapping).astype(float)
data_train['edjefa'] = data_train['edjefa'].apply(mapping).astype(float)
data_train['edjefe'] = data_train['edjefe'].apply(mapping).astype(float)

data_test['dependency'] = data_test['dependency'].apply(mapping).astype(float)
data_test['edjefa'] = data_test['edjefa'].apply(mapping).astype(float)
data_test['edjefe'] = data_test['edjefe'].apply(mapping).astype(float)
#converting into percentages
data_train['males_above_12'] = data_train['r4h2']/data_train['r4h3']
data_train['person_above_12'] = data_train['r4t2']/data_train['r4t3']
data_train['size_to_person_ratio'] = data_train['tamhog']/data_train['tamviv']

data_test['males_above_12'] = data_test['r4h2']/data_test['r4h3']
data_test['person_above_12'] = data_test['r4t2']/data_test['r4t3']
data_test['size_to_person_ratio'] = data_test['tamhog']/data_test['tamviv']
data_train['males-above_12'] = data_train['males_above_12'].fillna(0)
data_test['males-above_12'] = data_test['males_above_12'].fillna(0)

data_train = data_train.fillna(0)
data_test = data_test.fillna(0)
# Assigning ID to sub before dropping
submission = data_test[['Id']]
#dropping other useless columns
cols = ['Id','idhogar','SQBescolari','SQBage','SQBhogar_total','SQBedjefe','SQBhogar_nin','SQBovercrowding','SQBdependency','SQBmeaned','agesq']
data_train.drop(cols, axis = 1, inplace = True)
data_test.drop(cols, axis = 1, inplace = True)
# Feature Engineering
#creating the matrics of features
y = data_train.Target.values
data_train.drop('Target', axis =1, inplace = True)
data_train.drop('target_des', axis =1, inplace = True)
X = data_train.iloc[:,:].values
X_test = data_test.iloc[:,:].values
from sklearn.ensemble import RandomForestClassifier as RFC
classifier = RFC(n_estimators =25 , random_state = 0)
classifier.fit(X,y)
predict_result = classifier.predict(X_test).astype(int)
sub = pd.DataFrame({
    "Id" : submission['Id'],
    "Target" : predict_result
})
sub.to_csv('sample_submission.csv', index =False, encoding = 'utf-8')
sub.head()
