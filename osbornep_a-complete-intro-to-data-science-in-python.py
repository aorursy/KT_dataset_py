# Standard Data Analytic Packages
import pandas as pd
import numpy as np

# Graphing Packages
import matplotlib.pyplot as plt
import seaborn as sns
# Set graphic style and figure size
sns.set(rc={'figure.figsize':(11,8)})
sns.set_style("white")
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

#train = pd.read_csv('/Users/philiposborne/Documents/Other Projects/Competition-Kaggle Titanic/titanic/train.csv')
#test = pd.read_csv('/Users/philiposborne/Documents/Other Projects/Competition-Kaggle Titanic/titanic/test.csv')
train.head()
train.tail()
train.describe()
test.head()
train['Name'][0]
train.Name[0]
train[['Name','Sex']].head()
# First row, second column
train.iloc[0,3]
# First row, second and third column
train.iloc[0,[3,4]]
# First and second row, second and third column
train.iloc[[0,1],[3,4]]
# First to 10th row, second and third column
train.iloc[0:10,[3,4]]
# First to 10th row, second to fourth column
train.iloc[0:10,3:5]
train.iloc[0,3]
# Find those older than 70   
train[train['Age']>70].head()
# Find those older than 70 OR younger than or equal to 18 
train[(train['Age']>70) | (train['Age']<=18)].head()
# Find those older than or equal to 60 and Female --> only 4 passengers who are above 60 are female
train[(train['Age']>=60) & (train['Sex']=="female")].head()
train.describe()
train['Age'].unique()
train[np.isnan(train['Age'])==True].head()
train = train.dropna(subset=['Age'])
train.describe()
train['Cabin'].unique()
train['Cabin'] = train['Cabin'].fillna('unknown')
train['Embarked'] = train['Embarked'].fillna('unknown')

train.head()
print("The mean age of passengers is:", train['Age'].mean())
print("The max age of passengers is:", train['Age'].max())
print("The min age of passengers is:", train['Age'].min())

#https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.mean.html
#https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.max.html
#https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.min.html
print("The mean age of passengers is:", np.mean(train['Age']))
print("The max age of passengers is:", np.max(train['Age']))
print("The min age of passengers is:", np.min(train['Age']))

#https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.ndarray.mean.html
#https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.ndarray.max.html
#https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.ndarray.min.html
print("The mean age of passengers, rounded to 3 decimal places, is:", np.round( np.mean(train['Age'])  ,3))
#e.g. string combining

string_1 = "Hello"
string_2 = "World!"

print(string_1 + " " + string_2)
sns.scatterplot(train['PassengerId'], train['Age'])

# Remove some of the plot lines
sns.despine()

# Graph Titles. "\n" will create a new line in the text if needed
plt.title("A Scatter Plot Showing the Age of Each Passenger ID \n The Mean Age of Passengers is: " +  str(np.round( np.mean(train['Age'])  ,3)))
plt.xlabel("Passenger ID")
plt.ylabel("Age")

plt.show()
# Set all bars to green

age_min = np.min(train['Age'])
age_mean = np.mean(train['Age'] )
age_max = np.max(train['Age'])
                   
sns.barplot(['min', 'mean', 'max'],[age_min, age_mean, age_max] , color = 'g')

# Remove some of the plot lines
sns.despine()

plt.title("Summary Statistics of the Age Feature \n All Bars Set to Green")
plt.ylabel("Age Value")

plt.show()
# Use 'Blues' palette range

age_min = np.min(train['Age'])
age_mean = np.mean(train['Age'] )
age_max = np.max(train['Age'])
                   
sns.barplot(['min', 'mean', 'max'],[age_min, age_mean, age_max] , palette = 'hls')

# Remove some of the plot lines
sns.despine()

plt.title("Summary Statistics of the Age Feature \n Colors Set by Palette Scheme 'hls'")
plt.ylabel("Age Value")

plt.show()
# Manually set each bar's color using hex codes

age_min = np.min(train['Age'])
age_mean = np.mean(train['Age'] )
age_max = np.max(train['Age'])
                   
sns.barplot(['min', 'mean', 'max'],[age_min, age_mean, age_max] , palette = ["#808282", "#B4ED20", "#918BC3"])

# Remove some of the plot lines
sns.despine()

plt.title("Summary Statistics of the Age Feature \n Palette Colors Manually Defined with Hexcodes")
plt.ylabel("Age Value")

plt.show()
sns.boxplot(x="Sex", y="Age", data=train, palette = ['#0066cc','#cc99ff'])

# Remove some of the plot lines
sns.despine()

plt.title("Box Plot to Compare the Age of Male and Females on the Titanic")
plt.xlabel("Gender")
plt.ylabel("Age")
plt.show()

sns.boxplot(x="Sex", y="Age", data=train, palette = ['#0066cc','#cc99ff'])
sns.swarmplot(x="Sex", y="Age", data=train, color=".25", alpha = 0.7)

# Remove some of the plot lines
sns.despine()

plt.title("Box Plot to Compare the Age of Male and Females on the Titanic")
plt.xlabel("Gender")
plt.ylabel("Age")
plt.show()

print("Number of passengers that were male: " , len(train[train['Sex']=='male']))
print("Number of passengers that were female: " , len(train[train['Sex']=='female']))
sns.violinplot(x="Sex", y="Age", data=train, palette = ['#0066cc','#cc99ff'])

# Remove some of the plot lines
sns.despine()

plt.title("Box Plot to Compare the Age of Male and Females on the Titanic")
plt.xlabel("Gender")
plt.ylabel("Age")
plt.show()
sns.boxplot(x="Pclass", y="Age", data=train, hue="Sex", palette = ['#0066cc','#cc99ff'])

# Remove some of the plot lines
sns.despine()

plt.title("Box Plot to Compare the Age of Passenger Classes by Gender on the Titanic")
plt.xlabel("Passenger Class")
plt.ylabel("Age")
plt.show()

sns.boxplot(x="Pclass", y="Fare", data=train, hue="Sex", palette = ['#0066cc','#cc99ff'])

# Remove some of the plot lines
sns.despine()

plt.title("Box Plot to Compare the Fare of Passenger Classes by Gender on the Titanic")
plt.xlabel("Passenger Class")
plt.ylabel("Fare")
plt.show()

train['Pclass_2'] = np.where(train['Pclass']==1,"First", 
                    np.where(train['Pclass']==2,"Second", 
                    np.where(train['Pclass']==3,"Third",
                             "error")))
sns.scatterplot(train['Age'], train['Fare'], hue = train['Pclass_2'])

# Remove some of the plot lines
sns.despine()

# Graph Titles. "\n" will create a new line in the text if needed
plt.title("A Scatter Plot Showing the Age and Fare for each Passenger")
plt.xlabel("Age")
plt.ylabel("Fare")

plt.show()
sns.scatterplot(train['Age'], train['Fare'], hue = train['Pclass_2'])

# Remove some of the plot lines
sns.despine()

# Graph Titles. "\n" will create a new line in the text if needed
plt.title("A Scatter Plot Showing the Age and Fare for each Passenger")
plt.xlabel("Age")
plt.ylabel("Fare")
plt.ylim(0,100)
plt.show()
sns.distplot(train[train['Pclass_2']=="First"]["Fare"], label = "First Class")
sns.distplot(train[train['Pclass_2']=="Second"]["Fare"], label = "Second Class")
sns.distplot(train[train['Pclass_2']=="Third"]["Fare"], label = "Third Class")

# Remove some of the plot lines
sns.despine()


plt.title("Histrograms to Compare the Fare of Passenger Classes on the Titanic")
plt.xlabel("Passenger Class")
plt.ylabel("Fare")

plt.ylim(0,0.2)

plt.legend()

plt.show()

# Remind ourselves the features available
train.head()
train['counter'] = 1
train[['Sex','Age','Survived','counter']].groupby(['Sex','Age', 'Survived']).count()[0:5]
train[['Sex','Age','Survived','counter']].groupby(['Sex','Age','Survived']).sum()[0:5]
train[['Sex','Age','Survived','counter']].groupby(['Sex','Age', 'Survived']).count().reset_index(drop=False)[0:5]
train_gender_age_surv_count = train[['Sex','Age','Survived','counter']].groupby(['Sex','Age', 'Survived']).count().reset_index(drop=False)
train_gender_age_surv_count['prob'] = train_gender_age_surv_count['counter']/sum(train_gender_age_surv_count['counter'])
train_gender_age_count_SURVIVED = train_gender_age_surv_count[train_gender_age_surv_count['Survived']==1].reset_index(drop=True)
train_gender_age_count_DIED = train_gender_age_surv_count[train_gender_age_surv_count['Survived']==0].reset_index(drop=True)
train_gender_age_count_SURVIVED.head()
male_surv_prob = sum(train_gender_age_count_SURVIVED[train_gender_age_count_SURVIVED['Sex']=="male"]['prob'])
female_surv_prob = sum(train_gender_age_count_SURVIVED[train_gender_age_count_SURVIVED['Sex']=="female"]['prob'])



# Remove some of the plot lines
sns.despine()

sns.barplot( ['Male', 'Female'], [male_surv_prob,female_surv_prob], palette = ['#0066cc','#cc99ff'] )

plt.title("Overall Proportion of those that Survived by Gender")
plt.ylabel("Probability")
plt.show()

f, axes = plt.subplots(1, 2, sharey=True, sharex=True)
sns.barplot(train_gender_age_count_SURVIVED[train_gender_age_count_SURVIVED['Sex']=="male"]['Age'],
            train_gender_age_count_SURVIVED[train_gender_age_count_SURVIVED['Sex']=="male"]['counter'], 
            color = '#0066cc', alpha = 0.9, label = 'Male Survived', ax=axes[0] )


sns.barplot(train_gender_age_count_SURVIVED[train_gender_age_count_SURVIVED['Sex']=="female"]['Age'],
            train_gender_age_count_SURVIVED[train_gender_age_count_SURVIVED['Sex']=="female"]['counter'], 
            color = '#cc99ff', alpha = 0.9, label = 'Female Survived', ax=axes[1] )



# Fix axis labels so they are not overlapping, solution found here:
# https://stackoverflow.com/questions/38809061/remove-some-x-labels-with-seaborn/38809632
import matplotlib.ticker as ticker

ax = plt.gca()
ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
ax.xaxis.set_major_locator(ticker.MultipleLocator(base=20))

# Remove some of the plot lines
sns.despine()


for ax in axes:
    ax.set_ylabel("Probability of Survival")
    ax.set_xlabel("Age")
    ax.legend()

plt.suptitle("Number of Passengers that Survived by Age and Gender ")



plt.show()
train_gender_age_count = train[['Sex','Age','counter']].groupby(['Sex','Age']).count().reset_index(drop=False)
train_gender_age_count['prob'] = train_gender_age_count['counter']/sum(train_gender_age_count['counter'])
train_gender_age_count.head()
male_surv_prob = sum(train_gender_age_count_SURVIVED[train_gender_age_count_SURVIVED['Sex']=="male"]['prob'])
female_surv_prob = sum(train_gender_age_count_SURVIVED[train_gender_age_count_SURVIVED['Sex']=="female"]['prob'])

male_total_prob = sum(train_gender_age_count[train_gender_age_count['Sex']=="male"]['prob'])
female_total_prob = sum(train_gender_age_count[train_gender_age_count['Sex']=="female"]['prob'])



# Remove some of the plot lines
sns.despine()

sns.barplot( ['Male', 'Female'], [male_surv_prob,female_surv_prob], palette = ['#0066cc','#cc99ff'], alpha = 0.9 )
sns.barplot( ['Male', 'Female'], [male_total_prob,female_total_prob], palette = ['#0066cc','#cc99ff'], alpha = 0.4 )

plt.title("Overall Proportion of those that Survived by Gender Compared to the Total Populations")
plt.ylabel("Probability")
plt.show()

f, axes = plt.subplots(1, 2, sharey=True, sharex=True)
sns.barplot(train_gender_age_count_SURVIVED[train_gender_age_count_SURVIVED['Sex']=="male"]['Age'],
            train_gender_age_count_SURVIVED[train_gender_age_count_SURVIVED['Sex']=="male"]['counter'], 
            color = '#0066cc', alpha = 0.9, label = 'Male Survived', ax=axes[0] )

sns.barplot(train_gender_age_count[train_gender_age_count['Sex']=="male"]['Age'],
            train_gender_age_count[train_gender_age_count['Sex']=="male"]['counter'], 
            color = '#0066cc', alpha = 0.4, label = 'Male Total', ax=axes[0] )


sns.barplot(train_gender_age_count_SURVIVED[train_gender_age_count_SURVIVED['Sex']=="female"]['Age'],
            train_gender_age_count_SURVIVED[train_gender_age_count_SURVIVED['Sex']=="female"]['counter'], 
            color = '#9966ff', alpha = 0.9, label = 'Female Survived', ax=axes[1] )


sns.barplot(train_gender_age_count[train_gender_age_count['Sex']=="female"]['Age'],
            train_gender_age_count[train_gender_age_count['Sex']=="female"]['counter'], 
            color = '#cc99ff', alpha = 0.4, label = 'Female Total', ax=axes[1] )


# Fix axis labels so they are not overlapping, solution found here:
# https://stackoverflow.com/questions/38809061/remove-some-x-labels-with-seaborn/38809632
import matplotlib.ticker as ticker

ax = plt.gca()
ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
ax.xaxis.set_major_locator(ticker.MultipleLocator(base=20))

# Remove some of the plot lines
sns.despine()


for ax in axes:
    ax.set_ylabel("Probability of Survival")
    ax.set_xlabel("Age")
    ax.legend()

plt.suptitle("Number of Passengers that Survived by Age and Gender ")



plt.show()
train_gender_age_count.head(3)
train_gender_age_count_SURVIVED.head(3)
train_gender_age_count_2 = train_gender_age_count.merge(train_gender_age_count_SURVIVED[['Sex','Age','prob']], how = 'left', on = ['Sex','Age'])
train_gender_age_count_2 = train_gender_age_count_2.fillna(0)
train_gender_age_count_2.head(3)
prob_surv_given_age_gender = train_gender_age_count_2

prob_surv_given_age_gender['prob'] = train_gender_age_count_2['prob_y']/train_gender_age_count_2['prob_x']
prob_surv_given_age_gender.head(3)

f, axes = plt.subplots(1, 2, sharey=True, sharex=True)
sns.barplot(prob_surv_given_age_gender[prob_surv_given_age_gender['Sex']=="male"]['Age'],
            prob_surv_given_age_gender[prob_surv_given_age_gender['Sex']=="male"]['prob'], 
            color = '#0066cc', alpha = 0.9, label = 'Male Survived', ax=axes[0] )

sns.barplot(prob_surv_given_age_gender[prob_surv_given_age_gender['Sex']=="female"]['Age'],
            prob_surv_given_age_gender[prob_surv_given_age_gender['Sex']=="female"]['prob'], 
            color = '#9966ff', alpha = 0.9, label = 'Female Survived', ax=axes[1] )



# Fix axis labels so they are not overlapping, solution found here:
# https://stackoverflow.com/questions/38809061/remove-some-x-labels-with-seaborn/38809632
import matplotlib.ticker as ticker

ax = plt.gca()
ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
ax.xaxis.set_major_locator(ticker.MultipleLocator(base=20))

# Remove some of the plot lines
sns.despine()


for ax in axes:
    ax.set_ylabel("Probability of Survival")
    ax.set_xlabel("Age")
    ax.legend()

plt.suptitle("Probability that a Passenger Survived GIVEN Age and Gender by Conditional Probability")



plt.show()
pred_surv = prob_surv_given_age_gender[ prob_surv_given_age_gender['prob']>=0.5]
pred_died = prob_surv_given_age_gender[ prob_surv_given_age_gender['prob']<0.5]

pred_died.head()
male_surv_ages = pred_surv[pred_surv['Sex']=='male']['Age']
female_surv_ages = pred_surv[pred_surv['Sex']=='female']['Age']
male_surv_ages
test.head()
test['pred_1'] = np.where( (test['Sex']=='male') & (np.isin(test['Age'], male_surv_ages )), 1,
                   np.where( (test['Sex']=='female') & (np.isin(test['Age'], female_surv_ages )), 1, 0     ))

test.head()
PassId = [
892,893,894,895,896,897,898,899,900,901,902,903,904,905,906,907,908,909,910,911,912,913,914,915,916,917,918,919,920,921,922,923,
924,925,926,927,928,929,930,931,932,933,934,935,936,937,938,939,940,941,942,943,944,945,946,947,948,949,950,951,952,953,954,955,956,
957,958,959,960,961,962,963,964,965,966,967,968,969,970,971,972,973,974,975,976,977,978,979,980,981,982,983,984,985,986,987,988,989,
990,991,992,993,994,995,996,997,998,999,1000,1001,1002,1003,1004,1005,1006,1007,1008,1009,1010,1011,1012,1013,1014,1015,1016,1017,1018,
1019,1020,1021,1022,1023,1024,1025,1026,1027,1028,1029,1030,1031,1032,1033,1034,1035,1036,1037,1038,1039,1040,1041,1042,1043,1044,1045,
1046,1047,1048,1049,1050,1051,1052,1053,1054,1055,1056,1057,1058,1059,1060,1061,1062,1063,1064,1065,1066,1067,1068,1069,1070,1071,1072,
1073,1074,1075,1076,1077,1078,1079,1080,1081,1082,1083,1084,1085,1086,1087,1088,1089,1090,1091,1092,1093,1094,1095,1096,1097,1098,1099,
1100,1101,1102,1103,1104,1105,1106,1107,1108,1109,1110,1111,1112,1113,1114,1115,1116,1117,1118,1119,1120,1121,1122,1123,1124,1125,1126,
1127,1128,1129,1130,1131,1132,1133,1134,1135,1136,1137,1138,1139,1140,1141,1142,1143,1144,1145,1146,1147,1148,1149,1150,1151,1152,1153,
1154,1155,1156,1157,1158,1159,1160,1161,1162,1163,1164,1165,1166,1167,1168,1169,1170,1171,1172,1173,1174,1175,1176,1177,1178,1179,1180,
1181,1182,1183,1184,1185,1186,1187,1188,1189,1190,1191,1192,1193,1194,1195,1196,1197,1198,1199,1200,1201,1202,1203,1204,1205,1206,1207,
1208,1209,1210,1211,1212,1213,1214,1215,1216,1217,1218,1219,1220,1221,1222,1223,1224,1225,1226,1227,1228,1229,1230,1231,1232,1233,1234,
1235,1236,1237,1238,1239,1240,1241,1242,1243,1244,1245,1246,1247,1248,1249,1250,1251,1252,1253,1254,1255,1256,1257,1258,1259,1260,1261,
1262,1263,1264,1265,1266,1267,1268,1269,1270,1271,1272,1273,1274,1275,1276,1277,1278,1279,1280,1281,1282,1283,1284,1285,1286,1287,1288,
1289,1290,1291,1292,1293,1294,1295,1296,1297,1298,1299,1300,1301,1302,1303,1304,1305,1306,1307,1308,1309]

surv_label = [
0,1,0,0,1,1,1,1,1,0,0,0,1,0,1,1,0,0,0,1,0,1,1,1,1,0,1,0,0,0,0,0,1,0,1,0,1,0,1,1,1,0,0,0,1,0,1,0,1,1,0,0,1,1,0,0,0,
1,0,1,0,0,0,1,1,0,0,0,1,1,1,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,1,0,1,1,0,1,0,0,1,1,0,0,0,1,0,0,1,1,0,1,1,0,0,0,1,0,0,0,
0,0,1,0,0,1,0,1,0,1,1,0,1,0,0,0,1,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,1,1,1,0,0,1,1,1,0,0,1,0,0,1,1,0,0,0,
0,0,1,1,1,1,1,0,0,1,0,1,0,1,0,0,0,0,1,0,0,1,0,1,1,0,0,1,0,0,1,0,1,0,1,1,0,0,1,0,0,0,0,0,0,1,0,1,0,1,1,1,1,1,0,0,0,0,
1,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,1,0,1,1,0,0,0,0,0,0,1,1,1,0,1,0,1,0,0,0,1,0,0,0,1,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,
0,1,1,0,1,0,0,1,0,0,0,1,0,0,0,1,1,0,1,0,1,0,1,0,0,1,0,0,0,0,0,0,1,0,0,1,0,1,0,0,0,1,0,0,1,0,0,0,0,0,0,0,1,0,1,0,1,0,
1,0,1,1,0,0,0,1,0,1,0,1,0,0,1,1,0,1,0,0,0,1,1,0,1,1,0,1,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,1,1,0,1,0,0,1,0,1,0,1,
1,0,0,1,0,0,1,0,0,1,0,0,1
]

test_labelled = pd.DataFrame({'PassengerId':PassId,
                              'Survived':surv_label})

test_labelled.head()                            
test_labelled = test_labelled.merge(test[['PassengerId','pred_1']], how='left', on = 'PassengerId')
test_labelled.head(3)
# Export Predictions to csv for submission
submission_1 = test_labelled[['PassengerId','pred_1']]
submission_1.index = submission_1['PassengerId']
submission_1 = submission_1[['pred_1']]
submission_1.columns = ['Survived']
#submission_1.to_csv('...directory.../titanic/pred_1.csv')
pred_1_err = sum(abs(test_labelled['Survived'] - test_labelled['pred_1']))/len(test_labelled)
pred_1_acc = 1-pred_1_err

print("The Error of our first prediction using conditional probability is:", np.round(pred_1_err*100,2), "%" )
print("The Accuracy of our first prediction using conditional probability is:", np.round(pred_1_acc*100,2), "%" )

from sklearn.metrics import confusion_matrix

true = test_labelled['Survived']
pred = test_labelled['pred_1']

tn, fp, fn, tp = confusion_matrix(true, pred).ravel()
print("True Positive Count = ", tp)
print("False Positive Count = ", fp)
print("True Negative Count = ", tn)
print("False Negative Count = ", fn)
pred_1_prec = tp/(tp+fp)
pred_1_recall = tp/(tp+fn)

print("Prediction 1 Precision = ", np.round(pred_1_prec,4))
print("Prediction 1 Recall = ", np.round(pred_1_recall,4))

train.head()
train['pred_1'] = np.where( (train['Sex']=='male') & (np.isin(train['Age'], male_surv_ages )), 1,
                   np.where( (train['Sex']=='female') & (np.isin(train['Age'], female_surv_ages )), 1, 0     ))
train.head()
pred_1_train_err = sum(abs(train['Survived'] - train['pred_1']))/len(train)
pred_1_train_acc = 1-pred_1_train_err

print("The Training Error of our first prediction using conditional probability is:", np.round(pred_1_train_err*100,2), "%" )
print("The Training Accuracy of our first prediction using conditional probability is:", np.round(pred_1_train_acc*100,2), "%" )

train_overfit = train[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked', 'counter']]

ALL_Count = train_overfit.groupby(['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked']).count().reset_index(drop=False)
ALL_Count['prob'] = ALL_Count['counter']/sum(ALL_Count['counter'])
ALL_Count.head()
surv_Count = train_overfit[train_overfit['Survived']==1].groupby(['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked']).count().reset_index(drop=False)
surv_Count['prob'] = surv_Count['counter']/sum(ALL_Count['counter'])
surv_Count.head()
ALL_Count = ALL_Count.merge(surv_Count, how='left', on=['Survived', 'Pclass',	'Sex', 'Age', 'SibSp', 'Parch', 'Fare','Cabin','Embarked'])
ALL_Count[ALL_Count['Survived']==1].head()
ALL_Count_surv = ALL_Count[ALL_Count['Survived']==1]
ALL_Count_surv['prob_surv_given_ALL'] = ALL_Count_surv['prob_y']/ALL_Count_surv['prob_x']
ALL_Count_surv.head()
ALL_Count_surv['prob_surv_given_ALL'].unique()
test.head()
test_overfit = test.merge(ALL_Count_surv[['Pclass',	'Sex', 'Age', 'SibSp', 'Parch', 'Fare','Cabin','Embarked','prob_surv_given_ALL']], 
                          how='left', on = [ 'Pclass',	'Sex', 'Age', 'SibSp', 'Parch', 'Fare','Cabin','Embarked'])
test_overfit = test_overfit.fillna(0)
test_overfit.columns = [ 'PassengerId', 'Pclass','Name','Sex', 'Age', 'SibSp', 'Parch','Ticket' ,'Fare','Cabin','Embarked', 'pred_1', 'pred_overfit']
test_overfit.head()
test_labelled_2 = test_labelled.merge(test_overfit[['PassengerId','pred_overfit']], how='left', on = 'PassengerId')
test_labelled_2.head(3)
pred_overfit_err = sum(abs(test_labelled_2['Survived'] - test_labelled_2['pred_overfit']))/len(test_labelled_2)
pred_overfit_acc = 1-pred_overfit_err


print("--.--.--.--.--.--.--.-- Pred 1 --.--.--.--.--.--.--.--.--.--.--.--.--")
print("The Error of our first prediction using conditional probability is:", np.round(pred_1_err*100,2), "%" )
print("The Accuracy of our first prediction using conditional probability is:", np.round(pred_1_acc*100,2), "%" )

print("")

print("--.--.--.--.--.--.--.-- Overfitting Prediction --.--.--.--.--.--.--.--")
print("The Error of our first prediction using conditional probability is:", np.round(pred_overfit_err*100,2), "%" )
print("The Accuracy of our first prediction using conditional probability is:", np.round(pred_overfit_acc*100,2), "%" )

from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets

n_neighbors = 5

# we only take the first two features. We could avoid this ugly
# slicing by using a two-dim dataset
X = np.array(train[['Age','Fare']])
y = train['Survived']

h = .02  # step size in the mesh

# Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00'])


# we create an instance of Neighbours Classifier and fit the data.
clf = neighbors.KNeighborsClassifier(n_neighbors, weights='uniform')
clf.fit(X, y)
pred_2 = clf.predict(np.array(test[['Age','Fare']].dropna(subset=['Age','Fare'])))
pred_2[0:10]
test_no_na = test.dropna(subset=['Age','Fare'])
test_no_na['pred_2'] = pred_2
sns.scatterplot(test_no_na['Age'], test_no_na['Fare'], hue = test_no_na['pred_2'])

# Remove some of the plot lines
sns.despine()

# Graph Titles. "\n" will create a new line in the text if needed
plt.title("A Scatter Plot Showing the Age and Fare for each Passenger and the K-Nearest Neighbour Predicted Labels" )
plt.xlabel("Age")
plt.ylabel("Fare")
plt.ylim(0,100)
plt.show()
test_labelled = test_labelled.merge(test_no_na[['PassengerId','pred_2']], how='left', on = 'PassengerId')
# Set any missing values due to not having age or far to 0
test_labelled['pred_2'] = test_labelled['pred_2'].fillna(0)
test_labelled.head(3)
pred_2_err = sum(abs(test_labelled['Survived'] - test_labelled['pred_2']))/len(test_labelled)
pred_2_acc = 1-pred_2_err


print("--.--.--.--.--.--.--.-- Pred 1 --.--.--.--.--.--.--.--.--.--.--.--.--")
print("The Error of our first prediction using conditional probability is:", np.round(pred_1_err*100,2), "%" )
print("The Accuracy of our first prediction using conditional probability is:", np.round(pred_1_acc*100,2), "%" )

print("")

print("--.--.--.--.--.--.--.-- Pred 2 --.--.--.--.--.--.--.--")
print("The Error of our first prediction using conditional probability is:", np.round(pred_2_err*100,2), "%" )
print("The Accuracy of our first prediction using conditional probability is:", np.round(pred_2_acc*100,2), "%" )

train.head()
train['Sex_2'] = pd.factorize(train['Sex'])[0]
train['Cabin_2'] = pd.factorize(train['Cabin'])[0]
train['Embarked_2'] = pd.factorize(train['Embarked'])[0]

train['Pclass'] = train['Pclass'].astype(int)
              

train.head()
test['Sex_2'] = pd.factorize(test['Sex'])[0]
test['Cabin_2'] = pd.factorize(test['Cabin'])[0]
test['Embarked_2'] = pd.factorize(test['Embarked'])[0]

test['Pclass'] = test['Pclass'].astype(int)
    

train.head()
train[['Pclass','Sex_2','Age','SibSp','Parch','Fare','Cabin_2','Embarked_2']].dtypes
n_neighbors = 5

# we only take the first two features. We could avoid this ugly
# slicing by using a two-dim dataset
X = np.array(train[['Pclass','Sex_2','Age','SibSp','Parch','Fare','Cabin_2','Embarked_2']])
y = train['Survived']

h = .02  # step size in the mesh

# Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00'])


# we create an instance of Neighbours Classifier and fit the data.
clf = neighbors.KNeighborsClassifier(n_neighbors, weights='uniform')
clf.fit(X, y)
pred_3 = clf.predict(np.array(test[['Pclass','Sex_2','Age','SibSp','Parch','Fare','Cabin_2','Embarked_2']].dropna()))
pred_3[0:10]
test_no_na['pred_3'] = pred_3
test_labelled = test_labelled.merge(test_no_na[['PassengerId','pred_3']], how='left', on = 'PassengerId')
# Set any missing values due to not having age or far to 0
test_labelled['pred_3'] = test_labelled['pred_3'].fillna(0)
test_labelled.head(3)
pred_3_err = sum(abs(test_labelled['Survived'] - test_labelled['pred_3']))/len(test_labelled)
pred_3_acc = 1-pred_3_err


print("--.--.--.--.--.--.--.-- Pred 1 --.--.--.--.--.--.--.--.--.--.--.--.--")
print("The Error of our first prediction using conditional probability is:", np.round(pred_1_err*100,2), "%" )
print("The Accuracy of our first prediction using conditional probability is:", np.round(pred_1_acc*100,2), "%" )

print("")

print("--.--.--.--.--.--.--.-- Pred 2 --.--.--.--.--.--.--.--")
print("The Error of our first prediction using conditional probability is:", np.round(pred_2_err*100,2), "%" )
print("The Accuracy of our first prediction using conditional probability is:", np.round(pred_2_acc*100,2), "%" )

print("")

print("--.--.--.--.--.--.--.-- Pred 3 --.--.--.--.--.--.--.--")
print("The Error of our first prediction using conditional probability is:", np.round(pred_3_err*100,2), "%" )
print("The Accuracy of our first prediction using conditional probability is:", np.round(pred_3_acc*100,2), "%" )


import time

KNN_outputs = pd.DataFrame()
for i in range(1,10):
    
    start_time = time.time()

    n_neighbors = 5*i

    # we only take the first two features. We could avoid this ugly
    # slicing by using a two-dim dataset
    X = np.array(train[['Pclass','Sex_2','Age','SibSp','Parch','Fare','Cabin_2','Embarked_2']])
    y = train['Survived']

    h = .02  # step size in the mesh

    # Create color maps
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00'])


    # we create an instance of Neighbours Classifier and fit the data.
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights='uniform')
    clf.fit(X, y)
    pred_4 = clf.predict(np.array(test[['Pclass','Sex_2','Age','SibSp','Parch','Fare','Cabin_2','Embarked_2']].dropna()))
    test_no_na['pred_4'] = pred_4
    test_labelled_2 = test_labelled.merge(test_no_na[['PassengerId','pred_4']], how='left', on = 'PassengerId')
    # Set any missing values due to not having age or far to 0
    test_labelled_2['pred_4'] = test_labelled_2['pred_4'].fillna(0)

    pred_4_err = sum(abs(test_labelled_2['Survived'] - test_labelled_2['pred_4']))/len(test_labelled_2)
    pred_4_acc = 1-pred_4_err
    
    end_time = time.time()
    time_taken = end_time - start_time
    
    KNN_outputs = KNN_outputs.append(pd.DataFrame({'k':n_neighbors,
                                                   'Accuracy':pred_4_acc,
                                                   'Error':pred_4_err,
                                                   'Time':time_taken}, index = [i]))
    
KNN_outputs.head()
sns.lineplot(KNN_outputs['k'], KNN_outputs['Error'], label="error")

plt.title("Results of Increase k in K-NN")
plt.xlabel("k")
plt.ylabel("Error")
plt.show()
sns.lineplot(KNN_outputs['k'], KNN_outputs['Time'], label="time")

plt.title("Results of Increase k in K-NN")
plt.xlabel("k")
plt.ylabel("Time taken(s)")
plt.show()
import time

KNN_outputs = pd.DataFrame()
for i in range(1,10):
    
    start_time = time.time()

    n_neighbors = 50*i

    # we only take the first two features. We could avoid this ugly
    # slicing by using a two-dim dataset
    X = np.array(train[['Pclass','Sex_2','Age','SibSp','Parch','Fare','Cabin_2','Embarked_2']])
    y = train['Survived']

    h = .02  # step size in the mesh

    # Create color maps
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00'])


    # we create an instance of Neighbours Classifier and fit the data.
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights='uniform')
    clf.fit(X, y)
    pred_4 = clf.predict(np.array(test[['Pclass','Sex_2','Age','SibSp','Parch','Fare','Cabin_2','Embarked_2']].dropna()))
    test_no_na['pred_4'] = pred_4
    test_labelled_2 = test_labelled.merge(test_no_na[['PassengerId','pred_4']], how='left', on = 'PassengerId')
    # Set any missing values due to not having age or far to 0
    test_labelled_2['pred_4'] = test_labelled_2['pred_4'].fillna(0)

    pred_4_err = sum(abs(test_labelled_2['Survived'] - test_labelled_2['pred_4']))/len(test_labelled_2)
    pred_4_acc = 1-pred_4_err
    
    end_time = time.time()
    time_taken = end_time - start_time
    
    KNN_outputs = KNN_outputs.append(pd.DataFrame({'k':n_neighbors,
                                                   'Accuracy':pred_4_acc,
                                                   'Error':pred_4_err,
                                                   'Time':time_taken}, index = [i]))
    
sns.lineplot(KNN_outputs['k'], KNN_outputs['Error'], label="error")

plt.title("Results of Increase k in K-NN")
plt.xlabel("k")
plt.ylabel("Error")
plt.show()