import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
dataframe = pd.read_csv('/kaggle/input/pima-indians-diabetes-database/diabetes.csv')
dataframe = dataframe.dropna(axis=0)
y = dataframe.Outcome
features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
X = dataframe[features]
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

def get_model_and_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = RandomForestRegressor(max_leaf_nodes=max_leaf_nodes, random_state=1)
    model.fit(train_X, train_y)
    prediction = model.predict(val_X)
    return (model, mean_absolute_error(val_y, prediction))

model, mae = get_model_and_mae(24, train_X, val_X, train_y, val_y)
print(mae)
pregnancies = int(input('Number of times pregnant: '))
glucose = int(input('Plasma glucose concentration a 2 hours in an oral glucose tolerance test: '))
bp = int(input('Diastolic blood pressure (mm Hg): '))
skin_thickness = int(input('Triceps skin fold thickness (mm): '))
insulin = int(input('2-Hour serum insulin (mu U/ml): '))
bmi = float(input('Body mass index (weight in kg/(height in m)^2): '))
pedigree_function = float(input('Diabetes pedigree function: '))
age = int(input('Age (years): '))

data = pd.DataFrame({'Pregnancies' : [pregnancies], 'Glucose': [glucose], 'BloodPressure': [bp], 'SkinThickness': [skin_thickness], 'Insulin': [insulin], 'BMI': [bmi], 'DiabetesPedigreeFunction': [pedigree_function], 'Age': [age]})
print(f'You have {model.predict(data)[0] * 100} percent chance of getting diabetes')