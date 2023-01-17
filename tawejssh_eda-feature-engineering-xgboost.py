#Importing Libraries

import os

import pandas as pd

import seaborn as sns

import numpy as np

import matplotlib.pyplot as plt

import plotly.express as px

import plotly.graph_objects as go

import xgboost as xgb

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

import shap
shap.initjs()
plt.style.use('Solarize_Light2')
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv('/kaggle/input/insurance/insurance.csv')

data.head()
data.info()
plt.figure(figsize=(10,5))

heatmap = sns.heatmap(data.corr(), annot=True, fmt=".1f")

heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=12)

heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=12)

plt.title('Correlation Matrix', fontsize=18)

plt.show()
plt.figure(figsize=(12,5))

sns.distplot(data['age'], kde=True)

plt.title('Age Distribution')

plt.show()
plt.figure(figsize=(15,5))

sns.violinplot(data=data, x='sex', y='age', hue='smoker')

plt.title("Age Distributions Violinplot")

plt.show()
plt.figure(figsize=(15,5))

sns.violinplot(data=data, x='region', y='age')

plt.title('Age Distributions Violinplot')

plt.show()
plt.figure(figsize=(15,5))

sns.catplot(data=data, x='sex', hue='smoker', col='region', kind='count', col_wrap=2)

plt.suptitle('Patients Sex Distribution "" Countplots', fontsize=15)

plt.subplots_adjust(top=0.9)

plt.show()
fig = px.histogram(data, 

                   x='bmi', 

                   marginal='box', 

                   title='BMI Distribution')

fig.show()
fig = px.histogram(data, x='bmi', 

                   color='sex', 

                   marginal='box', 

                   title='BMI Distribution Over Patients Sex')

fig.show()
fig = px.histogram(data, x='bmi', 

                   color='region', 

                   marginal='box', 

                   title='BMI Distribution Over Patients Region')

fig.show()
fig = px.histogram(data, x='bmi', 

                   color='smoker', 

                   marginal='box', 

                   title='BMI Distribution Following Smoking Activity')

fig.show()
plt.figure(figsize=(15,5))

sns.countplot(data=data, x='children')

plt.title('Number Of Children Countplot')

plt.show()
plt.figure(figsize=(15,5))

sns.countplot(data=data, x='region', hue='children')

plt.title('Number of Children Distribution Over Regions')

plt.show()
plt.figure(figsize=(15,5))

sns.countplot(data=data, x='smoker')

plt.title("Smokers' Countplot")

plt.show()
plt.figure(figsize=(15,5))

sns.catplot(data=data,

            x='smoker', 

            y='age',

            hue='sex', 

            col='region', 

            kind='box', 

            col_wrap=2)

plt.suptitle('Smokers Age Distributions', fontsize=15)

plt.subplots_adjust(top=0.9)

plt.show()
plt.figure(figsize=(15,5))

sns.countplot(data=data, x='region')

plt.title("Regions' Countplot")

plt.show()
fig = px.histogram(data, 

                   x='charges', 

                   marginal='box', 

                   title='Charges Distribution')

fig.show()
fig = px.density_contour(data, x="age", y="charges",

                         facet_col="sex",

                         color='smoker',

                         marginal_x="histogram",

                         marginal_y="histogram")

fig.show()
fig = plt.figure(figsize=(11,5))

g = sns.jointplot(data=data, x='bmi', y='charges', kind='kde', color='g', height=9)

g.fig.suptitle('Charges as function of BMI', fontsize=15)

g.fig.subplots_adjust(top=0.95)

plt.show()
fig = px.scatter(data, x="bmi", y="charges", color="smoker",

                 size='charges')

fig.update_layout(title_text='charges = f(bmi)')

fig.show()
fig = go.Figure()



for n_children in data.children.unique():

    df = data.loc[data.children == n_children]

    fig.add_trace(go.Box(

        y=df.charges.values,

        name=str(n_children),

        boxpoints='all',

        jitter=0.5,

        whiskerwidth=0.2,

        #fillcolor=cls,

        marker_size=2,

        line_width=1)

    )

fig.update_layout(title_text="Charges Distribution Over Possible Numbers of Children")

fig.show()
fig = go.Figure()



for region in data.region.unique():

    df = data.loc[data.region == region]

    fig.add_trace(go.Box(

        y=df.charges.values,

        name=region,

        boxpoints='all',

        jitter=0.5,

        whiskerwidth=0.2,

        #fillcolor=cls,

        marker_size=2,

        line_width=1)

    )

fig.update_layout(title_text="Charges Distribution Over Regions")

fig.show()
new_data = data.copy()
new_data.loc[new_data['bmi']<25, 'bmi_class'] = 'Normal'

new_data.loc[(new_data['bmi']<30) & (new_data['bmi']>=25), 'bmi_class'] = 'Overweight'

new_data.loc[(new_data['bmi']<35) & (new_data['bmi']>=30), 'bmi_class'] = 'Class 1'

new_data.loc[(new_data['bmi']<40) & (new_data['bmi']>=35), 'bmi_class'] = 'Class 2'

new_data.loc[new_data['bmi']>=40, 'bmi_class'] = 'Class 3'
new_data.loc[(new_data['age']<31) & (new_data['age']>=18), 'age_class'] = 'Young'

new_data.loc[(new_data['age']<51) & (new_data['age']>=31), 'age_class'] = 'Adult'

new_data.loc[new_data['age']>=51, 'age_class'] = 'Old'
new_data.head()
bmi_classes_data = new_data.groupby('bmi_class').count()

fig = px.pie(values=bmi_classes_data['age'].values, names=bmi_classes_data['age'].index,

             title='BMI Classes Distribution')

fig.show()
age_classes_data = new_data.groupby('age_class').count()

fig = px.pie(values=age_classes_data['age'].values, names=age_classes_data['age'].index,

             title='Age Classes Distribution')

fig.show()
fig = px.histogram(new_data, x='charges', 

                   color='bmi_class', 

                   marginal='box', 

                   title='Charges Distribution Over BMI Classes')

fig.show()
fig = px.histogram(new_data, x='charges', 

                   color='age_class', 

                   marginal='box', 

                   title='Charges Distribution Over Age Classes')

fig.show()
encoder = LabelEncoder()

new_data['bmi_class'] = encoder.fit_transform(new_data['bmi_class'])

new_data['age_class'] = encoder.fit_transform(new_data['age_class'])

new_data['smoker'] = encoder.fit_transform(new_data['smoker'])

new_data['region'] = encoder.fit_transform(new_data['region'])

new_data['sex'] = encoder.fit_transform(new_data['sex'])
new_data['log_charges'] = np.log(new_data['charges'])
figure = plt.figure(figsize=(15,5))

ax1 = figure.add_subplot(2,1,1)

g1 = sns.distplot(new_data['charges'],ax=ax1)

ax2 = figure.add_subplot(2,1,2)

g2 = sns.distplot(new_data['log_charges'],ax=ax2)

figure.suptitle('Log Transformation of Charges Distribution', fontsize=15)

figure.tight_layout()

plt.subplots_adjust(top=0.9)

plt.show()
fig = px.histogram(new_data, 

                   x='charges', 

                   marginal='box', 

                   title='Initial Charges Distribution')

fig.show()





fig = px.histogram(new_data, 

                   x='log_charges', 

                   marginal='box', 

                   title='Log Transformation of Charges Distribution')

fig.show()
X, y = new_data.drop(columns=['charges', 'log_charges']), new_data['log_charges']

ground_truth = new_data['charges']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=44)
model = xgb.XGBRegressor(n_estimators=200,

                        learning_rate=0.1,

                        max_depth=10,

                        colsample_bytree=0.85,

                        reg_lambda=1)
model.fit(X_train, y_train)
fig, ax = plt.subplots(figsize=(13,5))

xgb.plot_importance(model, ax=ax, importance_type='cover')

plt.show()
importance_explainer = shap.TreeExplainer(model)
shap_values = importance_explainer.shap_values(X_train)
shap.summary_plot(shap_values, X_train, plot_type="bar")
shap.summary_plot(shap_values, X_train)
for feature in X_train.columns:

    shap.dependence_plot(feature, shap_values, X_train)
shap.force_plot(importance_explainer.expected_value, shap_values[0,:], X_train.iloc[0,:])
X_train.iloc[0,:], y_train.iloc[0]
shap.force_plot(importance_explainer.expected_value, shap_values, X_train)
predictions = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(predictions, y_test))

print(f'RMSE: {np.round(rmse, 3)}')
plt.figure(figsize=(20,6))

plt.plot(np.arange(len(predictions)), predictions, label='Predictions')

plt.plot(np.arange(len(y_test)), y_test, color='r', label='Ground Truth')

plt.ylabel('Log Charges')

plt.legend()

plt.show()