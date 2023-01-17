import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('seaborn')
data = pd.read_csv('../input/health-insurance-cross-sell-prediction/train.csv')

print(data.info())
data.head()
#Explore categorical values



fig, ax = plt.subplots(2, 3, figsize = (15,8))

fig.tight_layout(pad=4.0)



ax[0][0].bar(data.Gender.value_counts().index, data.Gender.value_counts())

ax[0][0].set_title('Gender')



ax[0][1].bar(data.Driving_License.value_counts().index, data.Driving_License.value_counts())

ax[0][1].set_title('Driving License')

ax[0][1].set_xticks([0,1])





ax[0][2].bar(data.Previously_Insured.value_counts().index, data.Previously_Insured.value_counts())

ax[0][2].set_title('Previously Insured')

ax[0][2].set_xticks([0,1])



ax[1][0].bar(data.Vehicle_Age.value_counts().index, data.Vehicle_Age.value_counts())

ax[1][0].set_title('Vehicle Age')



ax[1][1].bar(data.Vehicle_Damage.value_counts().index, data.Vehicle_Damage.value_counts())

ax[1][1].set_title('Vehicle_Damage')



ax[1][2].bar(data.Response.value_counts().index, data.Response.value_counts())

ax[1][2].set_title('Response')

ax[1][2].set_xticks([0,1])
#Correlation

plt.figure(figsize = (10, 10))

sns.heatmap(data.corr(), annot = True, fmt = '0.2g')
#data is highly imbalanced

response_1 = data[data.Response == 1]

response_0 = data[data.Response == 0]

data_new = data.append([response_1]*(len(response_0)//len(response_1) - 1))

print(data_new.Response.value_counts())

data_new = data_new.sample(frac = 1.0, random_state = 0)



num_cols = ['Age', 'Region_Code', 'Annual_Premium', 'Vintage', 'Policy_Sales_Channel']

categorical_col = ['Gender','Previously_Insured', 'Vehicle_Age', 'Vehicle_Damage']



imp_columns = num_cols + categorical_col



y = data_new.Response

x = data_new[imp_columns]



from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = 0.3, random_state = 0)
from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import MinMaxScaler

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import mean_absolute_error



#Categorical columns

preprocessor = ColumnTransformer(transformers = [('minmax',MinMaxScaler(), num_cols),('onehot', OneHotEncoder(), categorical_col)])



#model

model = RandomForestClassifier(random_state=0)



#model pipeline

model_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])



model_pipeline.fit(x_train, y_train)



predictions = model_pipeline.predict(x_val)



mae = mean_absolute_error(y_val, predictions)

print(mae)
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix, roc_curve, roc_auc_score



print('Accuracy = ',accuracy_score(y_val, predictions))

print('Recall = ',recall_score(y_val, predictions))

print('Precision = ',precision_score(y_val, predictions))



cm=confusion_matrix(y_val, predictions)



fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(13,5))



fpr, tpr, _ = roc_curve(y_val, predictions)

auc = roc_auc_score(y_val ,predictions)

ax1.plot([0,1],[0,1],linestyle='--')

ax1.plot(fpr,tpr,label="auc = %.5f"% auc)

ax1.legend(loc=4)



sns.heatmap(cm,annot=True,cmap='Blues',fmt='g')
#final running



#test data

test_data = pd.read_csv('../input/health-insurance-cross-sell-prediction/test.csv')



#fitting on whole train data

model_pipeline.fit(x, y)



#generating predictions

test_predictions = model_pipeline.predict(test_data[imp_columns])
#submission

#output = pd.DataFrame({'ID': test_data.id, 'Response': test_predictions})

#output.to_csv('submission.csv', index=False)

print("Submission saved")