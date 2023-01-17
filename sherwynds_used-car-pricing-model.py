import pandas as pd



data = pd.read_csv('../input/used-car-dataset-ford-and-mercedes/focus.csv')

print('Raw Data')

print(data.head())
transmission = pd.get_dummies(data['transmission'], drop_first=True)

fuelType = pd.get_dummies(data['fuelType'], drop_first=True)
data = data.drop(columns=['model', 'transmission', 'fuelType'])

data = data.join(transmission)

data = data.join(fuelType)

data.rename(columns={'year': 'year', 'price': 'price', 'mileage': 'mileage', 'engineSize': 'engine_size', 'Manual': 'manual', 'Semi-Auto': 'semi_auto', 'Petrol': 'petrol'}, inplace=True)

data = data[['year', 'mileage', 'engine_size', 'manual', 'semi_auto', 'petrol', 'price']]

print('Organized Data')

print(data.head())
import numpy as np

from sklearn.model_selection import train_test_split



X = data[['year', 'mileage', 'engine_size', 'manual', 'semi_auto', 'petrol']]

y = data['price']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

X_train = X_train.to_numpy()

X_test = X_test.to_numpy()

y_train = y_train.to_numpy()

y_test = y_test.to_numpy()



print('X_train\n' + str(X_train[:4,:]) + '\n')

print('y_train\n' + str(y_train[:4]) + '\n')

print('X_test\n' + str(X_test[:4,:]) + '\n')

print('y_test\n' + str(y_test[:4]))
# Select all rows from the second (mileage) column of the numpy arrays

X_train_mileage = X_train[:,1]

X_test_mileage = X_test[:,1]



print('X_train_mileage: ' + str(X_train_mileage))

print('y_train: ' + str(y_train))



from bokeh.plotting import figure, show

from bokeh.io import output_notebook

output_notebook()



train_plot = figure(title='Training Data', x_axis_label='Mileage', y_axis_label='Price')

train_plot.circle(x=X_train_mileage, y=y_train, color='blue')

show(train_plot)
from sklearn.preprocessing import PolynomialFeatures



cubic = PolynomialFeatures(degree=3)

X_train_mileage_cubic = cubic.fit_transform(X_train_mileage[:,None])

X_test_mileage_cubic = cubic.fit_transform(X_test_mileage[:,None])



print('X_train_mileage_cubic \n' + str(X_train_mileage_cubic[:4,:]))
from sklearn import linear_model



cubic_model = linear_model.LinearRegression()

cubic_model.fit(X_train_mileage_cubic, y_train)



train_predictions_cubic = cubic_model.predict(X_train_mileage_cubic)

test_predictions_cubic = cubic_model.predict(X_test_mileage_cubic)
from bokeh.layouts import row



train_plot_cubic = figure(title='Train Data', x_axis_label='Mileage', y_axis_label='Price')

test_plot_cubic = figure(title='Test Data', x_axis_label='Mileage', y_axis_label='Price')



train_plot_cubic.circle(x=X_train_mileage, y=y_train, color='blue', legend_label='Actual')

train_plot_cubic.circle(x=X_train_mileage, y=train_predictions_cubic, color='red', legend_label='Predicted')



test_plot_cubic.circle(x=X_test_mileage, y=y_test, color='blue', legend_label='Actual')

test_plot_cubic.circle(x=X_test_mileage, y=test_predictions_cubic, color='red', legend_label='Predicted')



show(row(train_plot_cubic,test_plot_cubic))



print('Train Data R-squared: ' + str(cubic_model.score(X_train_mileage_cubic,y_train)))

print('Test Data R-squred: ' + str(cubic_model.score(X_test_mileage_cubic,y_test)))
X_train_mileage_rl = X_train_mileage[:,None]

X_test_mileage_rl = X_test_mileage[:,None]



def addInvLogFeatures(numeric):

    log_feats = numeric.copy()

    valid = (log_feats != 1) & (log_feats > 0)

    log_feats[valid] = np.log(log_feats[valid]) / np.log(10)

    log_feats[log_feats <= 0] = 1e-10

    inv_log_feats = 1 / log_feats

    return np.hstack([numeric, inv_log_feats, numeric * inv_log_feats])



X_train_mileage_rl = addInvLogFeatures(X_train_mileage_rl)

X_test_mileage_rl = addInvLogFeatures(X_test_mileage_rl)



print('X_train_mileage_rl \n' + str(X_train_mileage_rl[:4,:]))
rl_model = linear_model.LinearRegression()

rl_model.fit(X_train_mileage_rl, y_train)



train_predictions_rl = rl_model.predict(X_train_mileage_rl)

test_predictions_rl = rl_model.predict(X_test_mileage_rl)
train_plot_rl = figure(title='Train Data', x_axis_label='Mileage', y_axis_label='Price')

test_plot_rl = figure(title='Test Data', x_axis_label='Mileage', y_axis_label='Price')



train_plot_rl.circle(x=X_train_mileage, y=y_train, color='blue', legend_label='Actual')

train_plot_rl.circle(x=X_train_mileage, y=train_predictions_rl, color='red', legend_label='Predicted')



test_plot_rl.circle(x=X_test_mileage, y=y_test, color='blue', legend_label='Actual')

test_plot_rl.circle(x=X_test_mileage, y=test_predictions_rl, color='red', legend_label='Predicted')



show(row(train_plot_rl,test_plot_rl))



print('Train Data R-squared: ' + str(rl_model.score(X_train_mileage_rl,y_train)))

print('Test Data R-squred: ' + str(rl_model.score(X_test_mileage_rl,y_test)))
from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler



X_train_numeric = X_train[:,:3]

X_test_numeric = X_test[:,:3]



scaler = StandardScaler()

X_train_numeric_scaled = scaler.fit_transform(X_train_numeric)

X_test_numeric_scaled = scaler.fit_transform(X_test_numeric)



pca = PCA(n_components=1)

X_train_pca = pca.fit_transform(X_train_numeric_scaled)



train_plot_pca = figure(title='Principal Component Analysis', x_axis_label='Principal Component', y_axis_label='price')

train_plot_pca.circle(x=X_train_pca[:,0], y=y_train, color='blue')

show(train_plot_pca)
X_train_cubic = cubic.fit_transform(X_train_numeric)

X_test_cubic = cubic.fit_transform(X_test_numeric)

# print("X_train_cubic \n" + str(X_train_cubic[:4,:]))
X_train_cubic_scaled = scaler.fit_transform(X_train_cubic)

X_test_cubic_scaled = scaler.fit_transform(X_test_cubic)



X_train_nominal = X_train[:,3:]

X_test_nominal = X_test[:,3:]



X_train_cubic_full = np.hstack([X_train_cubic_scaled, X_train_nominal])

X_test_cubic_full = np.hstack([X_test_cubic_scaled, X_test_nominal])



# print("X_train_cubic_full \n" + str(X_train_cubic_full[:4,:]))
cubic_model = linear_model.RidgeCV()

cubic_model.fit(X_train_cubic_full, y_train)



train_predictions_cubic = cubic_model.predict(X_train_cubic_full)

test_predictions_cubic = cubic_model.predict(X_test_cubic_full)
print('Train Data R-squared: ' + str(cubic_model.score(X_train_cubic_full,y_train)))

print('Test Data R-squred: ' + str(cubic_model.score(X_test_cubic_full,y_test)))
df = pd.DataFrame({"Predicted": test_predictions_cubic, "Actual": y_test})

df['% Difference'] = (abs(df['Predicted']-df['Actual'])/df['Actual'])*100



print("Percentage Difference between Predicted and Actual Values (Cubic Model)")

print(df.head())

print("\nMean % Difference between Predicted and Actual Values: " + str(df['% Difference'].mean()) +"%")
X_train_rl = addInvLogFeatures(X_train_numeric)

X_test_rl = addInvLogFeatures(X_test_numeric)

# print('X_train_rl \n' + str(X_train_rl[:4,:]))
X_train_rl_scaled = scaler.fit_transform(X_train_rl)

X_test_rl_scaled = scaler.fit_transform(X_test_rl)



X_train_rl_full = np.hstack([X_train_rl_scaled, X_train_nominal])

X_test_rl_full = np.hstack([X_test_rl_scaled, X_test_nominal])



# print("X_train_rl_full \n" + str(X_train_rl_full[:4,:]))
rl_model = linear_model.RidgeCV()

rl_model.fit(X_train_rl_full, y_train)



train_predictions_rl = rl_model.predict(X_train_rl_full)

test_predictions_rl = rl_model.predict(X_test_rl_full)
print('Train Data R-squared: ' + str(rl_model.score(X_train_rl_full,y_train)))

print('Test Data R-squred: ' + str(rl_model.score(X_test_rl_full,y_test)))
df = pd.DataFrame({"Predicted": test_predictions_rl, "Actual": y_test})

df['% Difference'] = (abs(df['Predicted']-df['Actual'])/df['Actual'])*100



print("Percentage Difference between Predicted and Actual Values (Reciprocal Log Model)")

print(df.head())

print("\nMean % Difference between Predicted and Actual Values: " + str(df['% Difference'].mean()) +"%")