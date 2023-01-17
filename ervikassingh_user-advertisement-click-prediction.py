import pandas as pd
data = pd.read_csv('../input/advertising.csv')
data.head()
import matplotlib.pyplot as plt

import seaborn as sns
sns.set()
plt.hist(x=data['Age'],label= 'Total Users')

plt.hist(x=data['Age'][data['Clicked on Ad']==1],label= 'Clicked on Add')

plt.xlabel('Age')

plt.title('AGE vs ADD CLICKS')

plt.legend()

plt.savefig('AGE vs ADD CLICKS.png')
gender_data=pd.crosstab(data['Male'],data['Clicked on Ad'])
gender_data
gender_data.rename({0:'Female',1:'Male'},axis=0,inplace=True)
gender_data
gender_data.plot(kind='bar')

plt.xlabel('Gender')

plt.title('GENDER vs ADD CLICKS')

plt.xticks(rotation=0)

plt.savefig('GENDER vs ADD CLICKS.png')
country_data=pd.crosstab(data['Country'],data['Clicked on Ad'])
country_data['total']=country_data.sum(axis=1)
country_data.head()
country_data=country_data.apply(lambda x:round(100*x/country_data['total']))
country_data.drop('total',axis=1,inplace=True)
country_data.head()
country_data= country_data.sort_values(1,ascending=False)
country_data.head(10)
country_data=country_data[:10].index.tolist()
country_data
plt.scatter(x=data['Daily Internet Usage'], y=data['Clicked on Ad'])

plt.title('DAILY USAGE vs ADD CLICKS')

plt.legend()

plt.savefig('DAILY USAGE vs ADD CLICKS.png')
training_data=data[['Age','Male','Daily Internet Usage']]

output_data=data['Clicked on Ad']
from sklearn.model_selection import train_test_split as tts

X,x_test,Y,y_test= tts(training_data,output_data,test_size=0.3,random_state=42)
from sklearn.linear_model import LogisticRegression

model=LogisticRegression()
model.fit(X,Y)
predictions=model.predict(x_test)
predictions[:5]
from sklearn.metrics import accuracy_score, confusion_matrix
score = accuracy_score(predictions,y_test)

score
test_data=pd.read_csv('../input/Test.csv')

test_data.head()
test_data=test_data[['Age','Male','Daily Internet Usage']]

test_data.head()
test_data.isnull().sum()
test_predictions=model.predict(test_data)
submission=pd.DataFrame({

    'Clicked on Ad':test_predictions

})
submission.head()
submission.to_csv('Prediction.csv')