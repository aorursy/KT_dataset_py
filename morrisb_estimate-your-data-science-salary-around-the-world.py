# To store data

import pandas as pd



# To do linear algebra

import numpy as np



# To create plots

import matplotlib.pyplot as plt



# To create interactive plots

import plotly.graph_objs as go

from plotly.offline import iplot, plot, init_notebook_mode

init_notebook_mode(True)



# To prepare training

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split



# To gbm light

from lightgbm import LGBMRegressor



# To optimize the hyperparameters of the model

from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, space_eval
# Load the dataset

df = pd.read_csv('../input/survey_results_public.csv', low_memory=False)



print('Shape Dataset:\t{}'.format(df.shape))

df.sample(3)
# Filter only good questions

df = df[['Country',

         'Employment',

         'CompanySize',

         'DevType',

         'YearsCoding',

         'ConvertedSalary',

         'Gender',

         'Age']]



print('Shape Dataset:\t{}'.format(df.shape))

df.sample(3)
# Split the jobs and count them

df_jobs = pd.DataFrame.from_records(df['DevType'].dropna().apply(lambda x: x.split(';')).values.tolist()).stack().reset_index(drop=True).value_counts()



# Create plot

df_jobs.plot(kind='barh', figsize=(10,7.5))

plt.title('Stack Overflow Survey Job-Count')

plt.xlabel('Job-Count')

plt.ylabel('Job')

plt.grid()

plt.show()
# Filter for the right jobs

df = df[~df['DevType'].isna()]

df = df[df['DevType'].str.contains('Data ')].drop('DevType', axis=1)



print('Shape Dataset:\t{}'.format(df.shape))

df.sample(3)
# Empty values

print('Empty Values:\t{}'.format(df['YearsCoding'].isna().sum()))



# Create subplots

fig, axarr = plt.subplots(2, figsize=(10,7.5))



# Create histogram

df['ConvertedSalary'].hist(bins=100, ax=axarr[0])

axarr[0].set_title('Salary Histogram')

axarr[0].set_xlabel('Salary')

axarr[0].set_ylabel('Count')



# Create sorted plot

df['ConvertedSalary'].sort_values().reset_index(drop=True).plot(ax=axarr[1])

axarr[1].set_title('Ordered Salaries')

axarr[1].set_xlabel('Ordered Index')

axarr[1].set_ylabel('Salary')



plt.tight_layout()

plt.show()
# Remove suspiciously low and high salaries

df = df[(df['ConvertedSalary']>1000) & (df['ConvertedSalary']<490000)]

print('Shape Dataset:\t{}'.format(df.shape))
# Top n countries

n = 20



# Empty values

print('Empty Values:\t{}'.format(df['YearsCoding'].isna().sum()))



# Create plot

df_country = df['Country'].value_counts().head(n)

df_country.plot(kind='barh', figsize=(10,7.5))

plt.title('Count For The Top {} Countries'.format(n))

plt.xlabel('Count')

plt.ylabel('Country')

plt.grid()

plt.show()
# Filter for the most frequent countries

df = df[ df['Country'].isin( df_country[:5].index ) ]



print('Shape Dataset:\t{}'.format(df.shape))
# Empty values

print('Empty Values:\t{}'.format(df['YearsCoding'].isna().sum()))



# Create plot

df['Employment'].value_counts().plot(kind='barh', figsize=(10,5))

plt.title('Kind Of Employment')

plt.xlabel('Count')

plt.ylabel('Employment')

plt.show()
# Impute and remove retired and unemployed colleagues

employment = ['Employed full-time', 

              'Employed part-time', 

              'Independent contractor, freelancer, or self-employed']

df = df[df['Employment'].fillna('Employed full-time').isin(employment)]



print('Shape Dataset:\t{}'.format(df.shape))
# Ordered company sacle

company_size = ['Fewer than 10 employees', '10 to 19 employees', '20 to 99 employees', '100 to 499 employees', '500 to 999 employees', '1,000 to 4,999 employees', '5,000 to 9,999 employees', '10,000 or more employees']



# Empty values

print('Empty Values:\t{}'.format(df['CompanySize'].isna().sum()))



# Create plot

df['CompanySize'].value_counts().reindex(company_size).plot(kind='barh', figsize=(10,7.5))

plt.title('Count Of Company Sizes')

plt.xlabel('Count')

plt.ylabel('Company Size')

plt.show()
# Create mapping for company size

mapping_company_size = {key:i for i, key in enumerate(company_size)}



# Drop empty values

df = df.dropna(subset=['CompanySize'])



# Transform category to numerical column

df['CompanySize'] = df['CompanySize'].map(mapping_company_size)



print('Shape Dataset:\t{}'.format(df.shape))
# Ordered years coding sacle

years_coding = ['0-2 years', '3-5 years', '6-8 years', '9-11 years', '12-14 years', '15-17 years', '18-20 years', '21-23 years', '24-26 years', '27-29 years', '30 or more years']



# Empty values

print('Empty Values:\t{}'.format(df['YearsCoding'].isna().sum()))



# Create plot

df['YearsCoding'].value_counts().reindex(years_coding).plot(kind='barh', figsize=(10,7.5))

plt.title('Count Of Years Coding')

plt.xlabel('Count')

plt.ylabel('Years Coding')

plt.show()
# Create mapping for years coding

mapping_years_coding = {key:i for i, key in enumerate(years_coding)}



# Transform category to numerical column

df['YearsCoding'] = df['YearsCoding'].map(mapping_years_coding)
# Empty values

print('Empty Values:\t{}'.format(df['Gender'].isna().sum()))



# Create plot

df['Gender'].value_counts().plot(kind='barh', figsize=(10,7.5))

plt.title('Gender Count')

plt.xlabel('Count')

plt.ylabel('Gender')

plt.show()
# Impute and map gender

df['Gender'] = df['Gender'].fillna('Male')

df = df[df['Gender'].isin(['Male', 'Female'])]

df['Gender'] = df['Gender'].map({'Male':0, 'Female':1})



print('Shape Dataset:\t{}'.format(df.shape))
# Ordered age sacle

age = ['Under 18 years old',

       '18 - 24 years old',

       '25 - 34 years old',

       '35 - 44 years old',

       '45 - 54 years old',

       '55 - 64 years old',

       '65 years or older']



# Empty values

print('Empty Values:\t{}'.format(df['Age'].isna().sum()))



# Create plot

df['Age'].value_counts().reindex(age).plot(kind='barh', figsize=(10,7.5))

plt.title('Age Count')

plt.xlabel('Count')

plt.ylabel('Age')

plt.show()
# Create mapping for years coding

mapping_age = {key:i for i, key in enumerate(age)}



# Transform category to numerical column

df['Age'] = df['Age'].fillna('25 - 34 years old').map(mapping_age)



print('Shape Dataset:\t{}'.format(df.shape))
# Create label

y = np.log(df['ConvertedSalary'].values)



# Create data

X = pd.get_dummies(df.drop('ConvertedSalary', axis=1)).values



# Create splitting of training and testing dataset

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=1)

X_valid, X_test, y_valid, y_test = train_test_split(X_valid, y_valid, test_size=0.25, random_state=1)



print('Training examples:\t\t{}\nExamples for optimization loss:\t{}\nFinal testing examples:\t\t{}'.format(X_train.shape[0], X_valid.shape[0], X_test.shape[0]))
# Define function to minimize

def objective(args):

    # Create & fit model

    model = LGBMRegressor(**args)

    model.fit(X_train, y_train)

    

    # Predict testset

    y_pred = model.predict(X_valid)

    

    # Compute rmse loss

    loss = np.sqrt(mean_squared_error(y_pred=y_pred, y_true=y_valid))

    return {'loss':loss, 'status':STATUS_OK, 'model':model}





# Setup search space

space = {'n_estimators': hp.choice('n_estimators', range(3000, 4500)),

         'max_depth': hp.choice('max_depth', range(25, 50)),

         'min_child_samples': hp.choice('min_child_samples', range(2, 10)),

         'reg_alpha': hp.uniform('reg_alpha', 0, 10),

         'reg_lambda': hp.uniform('reg_lambda', 1000, 3000)}





# Minmize function

trials = Trials()

best = fmin(objective, space, algo=tpe.suggest, max_evals=1000, trials=trials, rstate=np.random.RandomState(1))



# Compute final loss

model = trials.best_trial['result']['model']

y_pred = model.predict(X_test)

loss = np.sqrt(mean_squared_error(y_pred=y_pred, y_true=y_test))





# Print training results

for key, value in space_eval(space, best).items():

    print('{}:\t{}'.format(key, value))

print('\nTraining loss:\t{}'.format(trials.best_trial['result']['loss']))

print('\nFinal loss:\t{}'.format(loss))
# Create all possible answers

possibilities = []

for a in range(len(company_size)):

    for b in range(len(years_coding)):

        for c in range(len(['male', 'female'])):

            for d in range(len(age)):

                for e in range(len(['Canada', 'Germany', 'India', 'UK', 'US'])):

                    for f in range(len(['Fulltime', 'Parttime'])):

                        vector = np.zeros(11)

                        vector[0] = a

                        vector[1] = b

                        vector[2] = c

                        vector[3] = d

                        vector[4+e] = 1

                        vector[9+f] = 1

                        possibilities.append(vector)

possibilities = np.array(possibilities)



# Predict salaries for all answers

all_salaries = np.round(np.exp(model.predict(possibilities)), -2)



# Create data-structure for all salaries

df_plot = pd.DataFrame(possibilities, columns=['Size', 'Years', 'Gender', 'Age', 'Canada', 'Germany', 'India', 'UK', 'US', 'Full', 'Part'])

df_plot['Salary'] = all_salaries







# Create template for an interactive heatmap

def createHeatmap(x, y, x_axis, y_axis, x_label, y_label, filename):

    # Create hover texts & annotations

    def getAnotations(grid):

        hovertexts = []

        annotations = []

        for i, size in enumerate(y_axis):

            row = []

            for j, years in enumerate(x_axis):

                salary = grid[i, j]/1000

                row.append('Salary: {:.1f} k$/a<br>{}: {}<br>{}: {}<br>'.format(salary, y_label, size ,y_label, years))

                annotations.append(dict(x=years, y=size, text='{:.1f}'.format(salary), ax=0, ay=0, font=dict(color='#000000')))

            hovertexts.append(row)

        return hovertexts, annotations



    # Create traces

    data = []

    all_annotations = []

    # Iterate countries

    countries = ['US', 'UK', 'Germany', 'India', 'Canada']

    for i, country in enumerate(countries):

        # Get data

        grid = df_plot[df_plot[country]==1].pivot_table(index=y, columns=x, values='Salary', aggfunc='median').values

        # Get annotations

        hovertexts, annotations = getAnotations(grid)

        all_annotations.append(annotations)

        # Create trace

        trace = go.Heatmap(x = x_axis,

                           y = y_axis,

                           z = grid,

                           visible = True if i==0 else False,

                           text = hovertexts,

                           hoverinfo = 'text',

                           colorscale = 'Picnic',

                           colorbar = dict(title = 'Yearly<br>Salary',

                                           ticksuffix = '$'))

        data.append(trace)



    # Create buttons

    buttons = []

    # Iterate countries

    for i, country in enumerate(countries):

        label = country

        title = 'Median Salary Of A Data Scientist In {}'.format(country)

        visible = [False] * len(countries)

        visible[i] = True

        annotations = all_annotations[i]

        # Create button

        buttons.append(dict(label=label, method='update', args=[{'visible':visible},{'title':title, 'annotations':annotations}]))



    updatemenus = list([dict(type = 'dropdown',

                             active = 0,

                             buttons = buttons)])



    # Create layout

    layout = go.Layout(title = 'Median Salary Of A Data Scientist In {}'.format(countries[0]),

                       xaxis = dict(title = x_label,

                                    tickangle = -30),

                       yaxis = dict(title = y_label,

                                    tickangle = -30),

                       annotations = all_annotations[0],

                       updatemenus = updatemenus)



    # Create plot

    figure = go.Figure(data=data, layout=layout)

    plot(figure, filename=filename)

    iplot(figure)
x = 'Years'

y = 'Size'

x_axis = ['0-2', '3-5', '6-8', '9-11', '12-14', '15-17', '18-20', '21-23', '24-26', '27-29', '30 or more']

y_axis = ['<10',  '10-19', '20-99', '100-499', '500-999', '1,000-4,999', '5,000-9,999', '10,000<']

x_label = 'Years Coding'

y_label = 'Employees'



createHeatmap(x, y, x_axis, y_axis, x_label, y_label, 'Salary_Coding.html')
x = 'Age'

y = 'Size'

x_axis = ['Under 18', '18-24', '25-34', '35-44', '45-54', '55-64', '65 years or older']

y_axis = ['<10',  '10-19', '20-99', '100-499', '500-999', '1,000-4,999', '5,000-9,999', '10,000<']

x_label = 'Age'

y_label = 'Employees'



createHeatmap(x, y, x_axis, y_axis, x_label, y_label, 'Salary_Age.html')