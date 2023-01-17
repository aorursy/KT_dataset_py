import pandas as pd

import numpy as np

data = pd.read_csv('/kaggle/input/cardio-vascular-disease-detection/cardio_train.csv', delimiter=';')
data.head()
data.info()
data.describe()
data.isna().sum()
import seaborn as sns

import matplotlib.pyplot as plt
def plot_incidence(feature):

    cats = set(data[feature].values)

    

    xs = range(0, len(cats))

    ys_bar=[]

    ys_line = []

    

    for cat in cats:

        ys_bar.append(data[data[feature] == cat].shape[0])

        ys_line.append(data[(data[feature] == cat) & (data.cardio == 1)].shape[0]/data.shape[0] * 100)

    

    fig, ax = plt.subplots()

    

    ax.bar(xs, ys_bar, color='grey')



    ax2 = ax.twinx()

    ax2.plot(xs, ys_line, color='teal')

    

    ax.set_xticks(xs)

    ax.set_xticklabels(cats, rotation=90)

    ax.set_xlabel(feature)

    

    ax.set_ylabel('Frequency (n)')

    ax2.set_ylabel('Incidence (%)')

    

    fig.suptitle(f"Cardio incidence by {feature}")

    

    return plt.show()
cont_vars = ['age', 'height', 'weight', 'ap_hi', 'ap_lo']



# first I transform the age variable to years instead of days

data.age = data.age.apply(lambda x: x / 365)



cat_vars = ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']

for var in cat_vars:

    plot_incidence(var)
for var in cont_vars:

    _ = sns.boxplot(x='cardio', y=var, data=data)

    plt.show()
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegressionCV
# i've decided to engineer the bmi feature and remove the height and weight 

# height and weight in general aren't as informative as their ratio to another 

# i then code the bmi based on the standard categories

data['bmi'] = round(data.weight/data.height * 100, 2)

data['bmi_cat'] = pd.cut(data.bmi, pd.IntervalIndex.from_tuples([(0, 18.5), (18.5, 25), (25, 30), (30, 1000)]))

cat_vars.append('bmi_cat')



cont_vars.remove('height')

cont_vars.remove('weight')



# i also want to make age a categorical variable

data.age = data.age.apply(lambda x: round(x))

data['age_cat'] = pd.qcut(data.age, q=10, duplicates='drop', labels=[x for x in range(0, 10)])



cat_vars.append('age_cat')

cont_vars.remove('age')



# list for my final variables

final_vars = []

final_vars.extend(cont_vars)

final_vars.extend(cat_vars)
# split into train and test set

x_train, x_test, y_train, y_test = train_test_split(data[final_vars], data.cardio, test_size=.1, random_state=42)
# setting up a pipeline to transform categorical variables to one hot encoded variables and to scale continuous variables

ct = ColumnTransformer(transformers=[('onehot', OneHotEncoder(), cat_vars), ('scaler', StandardScaler(), cont_vars)])



# note that we only fit it using data that will be used to fit the model: x_train

ct.fit(x_train)
# transforming x_train and x_test according to pipeline

x_train = ct.transform(x_train)

x_test = ct.transform(x_test)
import time



lr = LogisticRegressionCV(cv=5, penalty='l1', solver='liblinear', refit=True, random_state=42)



start = time.time()

lr.fit(x_train, y_train.values)

end = time.time()

print(f"logistic regression fit in {(end - start) /60} mins")
from sklearn.metrics import classification_report



for key, value in {'TRAIN': [x_train, y_train], 'TEST': [x_test, y_test]}.items():

    preds = lr.predict(value[0])

    print(f"{key} RESULTS\n\n{classification_report(preds, value[1])}\n\n")
feature_names = []

feature_names.extend(cont_vars)



for cat in cat_vars:

    for val in set(data[cat].values):

        feature_names.append(f"{cat}_{val}")
feature_coefs = {feature: coefficient for feature, coefficient in zip(feature_names, lr.coef_[0])} 
feature_df = pd.Series(feature_coefs).to_frame()

feature_df = feature_df.reset_index()



feature_df.rename(columns={'index': 'feature', 0: 'log_prob'}, inplace=True)



feature_df['odds'] = feature_df.log_prob.apply(np.exp)
feature_df
# manually changing the odds ratio for age_cat_8 because it was stretching the plot too much

feature_df.at[28, 'odds'] = 4
# plotting

ys = [y for y in range(0, 30)]

xs = feature_df.odds.values

cs = []



for x in xs:

    if x < 1:

        cs.append('blue')

    elif x == 1:

        

        cs.append('grey')

    else:

        cs.append('orange')

    

fig = plt.figure(figsize=(8, 8))

_ = plt.scatter(xs, ys,s=30,color=cs)



plt.yticks(ticks=ys, labels=feature_df.feature.values)

plt.xlabel('Odds Ratio')

plt.ylabel('Feature')

plt.title('Odds Ratios for Features')

plt.show()