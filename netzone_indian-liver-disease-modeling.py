from IPython.display import HTML



HTML('''<script>

code_show=true; 

function code_toggle() {

 if (code_show){

 $('div.input').hide();

 } else {

 $('div.input').show();

 }

 code_show = !code_show

} 

$( document ).ready(code_toggle);

</script>

<form action="javascript:code_toggle()"><input type="submit" value="Toggle on/off the raw code."></form>''')
import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

from matplotlib import cm

%matplotlib inline



from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly.graph_objs as go



import seaborn as sns

sns.set(style="white", palette="muted", color_codes=True)



init_notebook_mode()
data = pd.read_csv("../input/indian_liver_patient.csv", low_memory=False)

data.head(2)
def missing_values_table(df): 

    mis_val = df.isnull().sum()

    mis_val_percent = 100 * df.isnull().sum()/len(df)

    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

    mis_val_table_ren_columns = mis_val_table.rename(columns = {0 : 'Missing Values', 1 : '% of Total Values'})

    return mis_val_table_ren_columns 

    

missing_values_table(data)
data.loc[data['Albumin_and_Globulin_Ratio'].isnull()]
data.describe()
sizing = data.groupby(['Dataset']).size().reset_index(name='Counts')

sizing['Percent'] = sizing['Counts']/sum(sizing['Counts']) * 100

sizing
gender_sizing = data.groupby(['Gender']).size().reset_index(name='Counts')

gender_sizing['Percent'] = gender_sizing['Counts']/sum(gender_sizing['Counts']) * 100

gender_sizing['Text'] = gender_sizing['Gender'] + ": " + round(gender_sizing['Percent'], 1).astype(str) + "%"
import plotly.tools as tls



fig = tls.make_subplots(rows=5, cols=2)

fig.append_trace({'y': data['Age'], 

                  'type': 'box', 

                  'name': 'Total Age'}, 1, 1)

fig.append_trace({'x': gender_sizing['Gender'], 'y': gender_sizing['Counts'], 

                  'type': 'bar', 'text': gender_sizing['Text'],

                  'name': 'Gender', 'marker': dict(color=['red', 'blue'])}, 1, 2)

fig.append_trace({'y': data['Total_Bilirubin'], 

                  'type': 'box', 'name': 'Total Bilirubin'}, 2, 1)

fig.append_trace({'y': data['Direct_Bilirubin'], 

                  'type': 'box', 'name': 'Direct Bilirubin'}, 2, 2)

fig.append_trace({'y': data['Alkaline_Phosphotase'], 

                  'type': 'box', 'name': 'Alkaline Phosphotase'}, 3, 1)

fig.append_trace({'y': data['Alamine_Aminotransferase'], 

                  'type': 'box', 'name': 'Alamine Aminotransferase'}, 3, 2)

fig.append_trace({'y': data['Aspartate_Aminotransferase'], 

                  'type': 'box', 'name': 'Aspartate Aminotransferase'}, 4, 1)

fig.append_trace({'y': data['Total_Protiens'], 

                  'type': 'box', 'name': 'Total Proteins'}, 4, 2)

fig.append_trace({'y': data['Albumin'], 

                  'type': 'box', 'name': 'Albumin'}, 5, 1)

fig.append_trace({'y': data['Albumin_and_Globulin_Ratio'], 

                  'type': 'box', 'name': 'Ratio of Albumin & Globulin'}, 5, 2)

    

fig['layout'].update(height=800, width=700, title='All features Visualizations')

iplot(fig, filename='all-features-visualization')    
corr = data[data.columns].corr()
sns.heatmap(corr, annot = True)
data = data.where(pd.notnull(data), data.median(), axis='columns')

data.isnull().sum()
from sklearn import preprocessing

col_name = data.iloc[:, 2:len(data.columns)-1].columns.tolist()

min_max_scaler = preprocessing.MinMaxScaler()

np_scaled = min_max_scaler.fit_transform(data.iloc[:, 2:len(data.columns)-1])

df_normalized = pd.DataFrame(np_scaled)

df_normalized.columns = col_name



# One hot encoding and drop the first column to prevent dummy trap variable

gender_df = data.iloc[:,1]

gender_df = pd.get_dummies(gender_df, prefix='Gender', drop_first=True)
my_df = pd.concat([gender_df, data.iloc[:, 0], df_normalized, data.iloc[:, len(data.columns)-1]], axis = 1)
fig = tls.make_subplots(rows=5, cols=2)

fig.append_trace({'y': my_df['Age'], 

                  'type': 'box', 

                  'name': 'Total Age'}, 1, 1)

fig.append_trace({'x': my_df['Gender_Male'],  

                  'type': 'histogram', 

                  'name': 'Gender', 'marker': dict(color=['red', 'blue'])}, 1, 2)

fig.append_trace({'y': my_df['Total_Bilirubin'], 

                  'type': 'box', 'name': 'Total Bilirubin'}, 2, 1)

fig.append_trace({'y': my_df['Direct_Bilirubin'], 

                  'type': 'box', 'name': 'Direct Bilirubin'}, 2, 2)

fig.append_trace({'y': my_df['Alkaline_Phosphotase'], 

                  'type': 'box', 'name': 'Alkaline Phosphotase'}, 3, 1)

fig.append_trace({'y': my_df['Alamine_Aminotransferase'], 

                  'type': 'box', 'name': 'Alamine Aminotransferase'}, 3, 2)

fig.append_trace({'y': my_df['Aspartate_Aminotransferase'], 

                  'type': 'box', 'name': 'Aspartate Aminotransferase'}, 4, 1)

fig.append_trace({'y': my_df['Total_Protiens'], 

                  'type': 'box', 'name': 'Total Proteins'}, 4, 2)

fig.append_trace({'y': my_df['Albumin'], 

                  'type': 'box', 'name': 'Albumin'}, 5, 1)

fig.append_trace({'y': my_df['Albumin_and_Globulin_Ratio'], 

                  'type': 'box', 'name': 'Ratio of Albumin & Globulin'}, 5, 2)

    

fig['layout'].update(height=800, width=700, title='All features Visualizations after normalization')

iplot(fig, filename='all-norm-features-visualization')  
my_df.head(2)
sns.heatmap(my_df[my_df.columns[:len(my_df.columns)-1]].corr(),annot=True,cmap='RdYlGn')

fig=plt.gcf()

fig.set_size_inches(10,8)

plt.show();
from sklearn.model_selection import train_test_split, KFold

from sklearn import svm

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics
train, test = train_test_split(my_df,

                            test_size = 0.3,

                            random_state = 0,

                            stratify = my_df['Dataset'])

train_X = train[train.columns[:len(train.columns)-1]]

test_X = test[test.columns[:len(test.columns)-1]]

train_Y = train['Dataset']

test_Y = test['Dataset'] 
types=['rbf','linear', 'sigmoid']

for i in types:

    model = svm.SVC(kernel=i, random_state=0)

    model.fit(train_X,train_Y)

    prediction = model.predict(test_X)

    print('Accuracy for SVM kernel =',i,'is',metrics.accuracy_score(prediction,test_Y))

    print('F1 score for SVM kernel =', i, ' is ', metrics.f1_score(prediction, test_Y))
model = LogisticRegression()

model.fit(train_X,train_Y)

prediction=model.predict(test_X)

print('The accuracy of the Logistic Regression is',metrics.accuracy_score(prediction,test_Y))

print('The F1 score of the Logistic Regression is',metrics.f1_score(prediction,test_Y))
model=DecisionTreeClassifier()

model.fit(train_X,train_Y)

prediction=model.predict(test_X)

print('The accuracy of the Decision Tree is',metrics.accuracy_score(prediction,test_Y))

print('The F1 score of the Decision Tree is',metrics.f1_score(prediction,test_Y))
model = RandomForestClassifier(n_estimators=100,random_state=0)

model.fit(train_X, train_Y)

prediction = model.predict(test_X)

print('The accuracy of the Random Forest is',metrics.accuracy_score(prediction,test_Y))

print('The F1 score of the Random Forest is',metrics.f1_score(prediction,test_Y))



print(pd.Series(model.feature_importances_,index=train_X.columns).sort_values(ascending=False))