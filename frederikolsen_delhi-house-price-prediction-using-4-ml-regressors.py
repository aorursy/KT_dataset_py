# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# load in the delhi house price dataframe
df = pd.read_csv('../input/delhi-house-price-prediction/MagicBricks.csv')
# inspect the first few rows of the dataframe
df.head()
# describe the dataset
df.describe()
# obtain datatypes
print(df.dtypes)
# outline the shape of the dataset
print(df.shape)
# import seaborn and matplotlib.pyplot for data visualisation
import seaborn as sns
import matplotlib.pyplot as plt

# produce a correlation matrix between variables where applicable
corr = df.corr()

# generate a custom diverging colormap
cmap = sns.diverging_palette(150, 275, as_cmap=True)

# draw the heatmap with the mask and correct aspect ratio
sns.set(rc={'figure.figsize': (17.0, 8.0)}, font_scale=1.2)
sns.heatmap(corr, cmap=cmap, vmax=1.0,vmin=min(corr.min()), center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)

plt.title("A correlation matrix to show the Pearson's \ncorrelation between variables", fontsize=20)
plt.yticks(rotation=0)
plt.xticks(rotation=45)
# Set up the matplotlib figure
f, ax = plt.subplots(3, 2, constrained_layout=True)
ax[0,0].hist(df['Area'], color='red', bins=50, edgecolor='black', linewidth=1.2)
ax[0,1].hist(df['BHK'], color='blue', bins=10, edgecolor='black', linewidth=1.2)
ax[1,0].hist(df['Bathroom'], color='green', bins=10, edgecolor='black', linewidth=1.2)
ax[1,1].hist(df['Parking'], color='purple', bins=50, edgecolor='black', linewidth=1.2)
ax[2,0].hist(df['Price'], color='yellow', bins=50, edgecolor='black', linewidth=1.2)
ax[2,1].hist(df['Per_Sqft'], color='pink', bins=50, edgecolor='black', linewidth=1.2)

# set all y-labels
plt.setp(ax[:, :], ylabel='Frequency')

# individually set subplot titles and x axis labels
plt.setp(ax[0, 0], xlabel='Area')
ax[0, 0].set_title('Area value frequency histogram', fontsize=20)

plt.setp(ax[0, 1], xlabel='BHK')
ax[0, 1].set_title('BHK value frequency histogram', fontsize=20)


plt.setp(ax[1, 0], xlabel='Bathroom')
ax[1, 0].set_title('Bathroom value frequency histogram', fontsize=20)


plt.setp(ax[1, 1], xlabel='Parking')
ax[1, 1].set_title('Parking value frequency histogram', fontsize=20)


plt.setp(ax[2, 0], xlabel='Price')
ax[2, 0].set_title('Price value frequency histogram', fontsize=20)


plt.setp(ax[2, 1], xlabel='Per_Sqft')
ax[2, 1].set_title('Per_Sqft value frequency histogram', fontsize=20)

f.tight_layout(pad=1.2)
# remove rows if they contain NA values
df = df.dropna()
# convert integers to floats
df['BHK'] = df['BHK'].astype(float)
df['Price'] = df['Price'].astype(float)
# get dummies for one hot encoding
furn_dummies = pd.get_dummies(df['Furnishing'], dtype=float)
loca_dummies = pd.get_dummies(df['Locality'], dtype=float)
stat_dummies = pd.get_dummies(df['Status'], dtype=float)
tran_dummies = pd.get_dummies(df['Transaction'], dtype=float)
type_dummies = pd.get_dummies(df['Type'], dtype=float)

# remove old columns
df = df.drop(['Furnishing', 'Locality', 'Status', 'Transaction', 'Type'], axis=1)

# concat the one hot encoded dataframes onto the main dataframe
df = pd.concat([df, furn_dummies, loca_dummies, stat_dummies, tran_dummies, type_dummies], axis=1)
print(df.shape)
# import the min-max scaler
from sklearn.preprocessing import MinMaxScaler

# define our scaler
scaler = MinMaxScaler(feature_range=(1, 2))

# no reverse transformation required with these columns, so we can use a fit_transform()
df['Area'] = scaler.fit_transform(np.expand_dims(np.log(df['Area']), axis=1))
df['BHK'] = scaler.fit_transform(np.expand_dims(df['BHK'], axis=1))
df['Parking'] = scaler.fit_transform(np.expand_dims(np.log(df['Parking']), axis=1))
df['Per_Sqft'] = scaler.fit_transform(np.expand_dims(np.log(df['Per_Sqft']), axis=1))

# we will ned to reverse transform the price values for evaluating the models,
# so we will separately define a scaler and store it as a variable for this variable
price_scaler = scaler.fit(np.expand_dims(np.log(df['Price']), axis=1))
df['Price'] = price_scaler.transform(np.expand_dims(np.log(df['Price']), axis=1))
# import the train_test_split function
from sklearn.model_selection import train_test_split

X_values = df.drop(['Price'], axis=1)
y_values = df['Price']

# now split our x and y values into train/test sets with an 80/20 percentage split
X_train, X_test, y_train, y_test = train_test_split(X_values, y_values, test_size=0.2)
print("X_train shape is", X_train.shape)
print("X_test shape is", X_test.shape)
print("y_train shape is", y_train.shape)
print("y_test shape is", y_test.shape)
# import sklearn's Support Vector Regression model
from sklearn.svm import SVR

# define the model
sv_regressor = SVR(kernel='linear')  # linear kernel achieved best results

# fit the model
sv_regressor.fit(X_train, y_train)
# import the r2 and mse evaluation metrics
from sklearn.metrics import r2_score, mean_squared_error

# make predictions with unseen testing data
sv_preds = sv_regressor.predict(X_test)

# calculate r-squared on inverse transformed data
sv_r2 = r2_score(np.exp(price_scaler.inverse_transform(np.expand_dims(y_test, axis=1))),
                 np.exp(price_scaler.inverse_transform(np.expand_dims(sv_preds, axis=1))))

# calculate mse on inverse transformed data
sv_mse = mean_squared_error(np.exp(price_scaler.inverse_transform(np.expand_dims(y_test, axis=1))),
                 np.exp(price_scaler.inverse_transform(np.expand_dims(sv_preds, axis=1))))

print('R-Squared: ', sv_r2)
print('Mean Squared Error: ', np.format_float_scientific(np.asarray(sv_mse), precision=3))
# load in the Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor

# define the model
rf_regressor = RandomForestRegressor(n_estimators=120)  # 120 estimators optimised performance

# fit the model
rf_regressor.fit(X_train, y_train)
# make predictions with unseen testing data
rf_preds = rf_regressor.predict(X_test)

# calculate r-squared on inverse transformed data
rf_r2 = r2_score(np.exp(price_scaler.inverse_transform(np.expand_dims(y_test, axis=1))),
                 np.exp(price_scaler.inverse_transform(np.expand_dims(rf_preds, axis=1))))

# calculate mse on inverse transformed data
rf_mse = mean_squared_error(np.exp(price_scaler.inverse_transform(np.expand_dims(y_test, axis=1))),
                 np.exp(price_scaler.inverse_transform(np.expand_dims(rf_preds, axis=1))))

print('R-Squared: ', rf_r2)
print('Mean Squared Error: ', np.format_float_scientific(np.asarray(rf_mse), precision=3))
# load in the Random Forest Regressor
from sklearn.neighbors import KNeighborsRegressor

# define the model
kn_regressor = KNeighborsRegressor(n_neighbors=4, algorithm='auto')  # 4 neighbours and 'auto' algorithm optimum

# fit the model
kn_regressor.fit(X_train, y_train)
# make predictions with unseen testing data
kn_preds = kn_regressor.predict(X_test)

# calculate r-squared on inverse transformed data
kn_r2 = r2_score(np.exp(price_scaler.inverse_transform(np.expand_dims(y_test, axis=1))),
                 np.exp(price_scaler.inverse_transform(np.expand_dims(kn_preds, axis=1))))

# calculate mse on inverse transformed data
kn_mse = mean_squared_error(np.exp(price_scaler.inverse_transform(np.expand_dims(y_test, axis=1))),
                 np.exp(price_scaler.inverse_transform(np.expand_dims(kn_preds, axis=1))))

print('R-Squared: ', kn_r2)
print('Mean Squared Error: ', np.format_float_scientific(np.asarray(kn_mse), precision=3))
import torch
import torch.nn as nn

# tensorize our x/y train/test data to form pytorch tensors
X_train_tensor = torch.from_numpy(X_train.to_numpy()).float()
X_test_tensor = torch.from_numpy(X_test.to_numpy()).float()
y_train_tensor = torch.from_numpy(y_train.to_numpy()).float()
y_test_tensor  = torch.from_numpy(y_test.to_numpy()).float()
print("X_train_tensor shape is", X_train_tensor.shape)
print("X_test_tensor shape is", X_test_tensor.shape)
print("y_train_tensor shape is", y_train_tensor.shape)
print("y_test_tensor shape is", y_test_tensor.shape)
# construct the deep learning MLP regressor
class MLP_Regressor(nn.Module):
    def __init__(self, input_dim, layer_sizes, dropout):
        super(MLP_Regressor, self).__init__()
        self.mlp = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, layer_sizes[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(layer_sizes[0], layer_sizes[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(layer_sizes[1], 1),
        )        
        
    # the forward pass through the network
    def forward(self, input_tensor):
        
        output_tensor = self.mlp(input_tensor)  # pass the input tensor through the mlp
        
        return output_tensor
    
# now lets define the model
mlp_regressor = MLP_Regressor(X_train_tensor.shape[1],
                               [4000, 1000],
                               0.02)
print(mlp_regressor)
loss_function = nn.MSELoss()  # mse loss function
optimizer = torch.optim.Adam(mlp_regressor.parameters(), lr=0.0001)  # adam's optimiser
epochs = 1000  # number of epochs
loss_vals_train = []  # hold the training loss values
loss_vals_valid = []  # hold the validation loss values

for i in range(epochs):
    y_pred_tensor = mlp_regressor(X_train_tensor)  # obtain y predictions
    single_loss = loss_function(y_pred_tensor[:-20], torch.unsqueeze(y_train_tensor[:-20], dim=1))  # calculate training loss
    loss_vals_train.append(single_loss.item())
    
    # now calculate the validation loss
    with torch.no_grad():  # disable the autograd engine
        val_loss = loss_function(y_pred_tensor[-20:], torch.unsqueeze(y_train_tensor[-20:], dim=1))  # calculate validation loss
        loss_vals_valid.append(val_loss.item())
    
    optimizer.zero_grad()  # zero the gradients
    single_loss.backward()  # backpropagate through the model
    optimizer.step()  # update parameters
    
    if i%25 == 0:
        print(f'epoch: {i:5} training loss: {single_loss.item():10.8f} validation loss: {val_loss.item():10.8f}')
sns.set(rc={'figure.figsize': (45.0, 20.0)})
sns.set(font_scale=8.0)
sns.set_context("notebook", font_scale=5.5, rc={"lines.linewidth": 1.0})
x_vals = np.arange(0, epochs, 1)
ax = sns.lineplot(x=x_vals, y=loss_vals_train)
ax = sns.lineplot(x=x_vals, y=loss_vals_valid)
ax.set_ylabel('Loss', labelpad=20, fontsize=75)
ax.set_xlabel('Epochs', labelpad=20, fontsize=75)
plt.legend(labels=['Training loss', 'Validation loss'], facecolor='white', framealpha=1)
plt.show()
# activate the evaluation mode for the model
mlp_regressor.eval()

# make predictions with the mlp model
mlp_preds = mlp_regressor(X_test_tensor).detach().numpy()

# calculate r-squared on inverse transformed data
mlp_r2 = r2_score(np.exp(price_scaler.inverse_transform(np.expand_dims(y_test, axis=1))),
                 np.exp(price_scaler.inverse_transform(mlp_preds)))

# calculate mse on inverse transformed data
mlp_mse = mean_squared_error(np.exp(price_scaler.inverse_transform(np.expand_dims(y_test, axis=1))),
                 np.exp(price_scaler.inverse_transform(mlp_preds)))

print('R-Squared: ', mlp_r2)
print('Mean Squared Error: ', np.format_float_scientific(np.asarray(mlp_mse), precision=3))
# store revelant information as lists
r2_vals = [sv_r2, rf_r2, kn_r2, mlp_r2]
mse_vals = [sv_mse, rf_mse, kn_mse, mlp_mse]
models = ['Support Vector \nRegression', 'Random Forest \nRegression',
          'K-Nearest Neighbour \nRegression', 'Deep Learning MLP \nRegression']

# convert lists into a single dataframe
accuracy_df = pd.DataFrame({'Model': models, 'R-Squared': r2_vals, 'MSE': mse_vals})
# plot r2 score as a barplot
sns.set_context("notebook", font_scale=4.5, rc={"lines.linewidth": 0.5})
ax = sns.barplot(x="Model", y="R-Squared", data=accuracy_df)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, horizontalalignment='right')
ax.set_ylabel('R-Squared', labelpad=50, fontsize=85)
ax.set_xlabel('Model', labelpad=50, fontsize=85)

plt.title("A Barplot comparing the testing data R-Squared \nof all four regressors", fontsize=100)
plt.show()
# plot MSE score as a barplot
sns.set_context("notebook", font_scale=4.5, rc={"lines.linewidth": 0.5})
ax = sns.barplot(x="Model", y="MSE", data=accuracy_df)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, horizontalalignment='right')
ax.set_ylabel('MSE', labelpad=50, fontsize=85)
ax.set_xlabel('Model', labelpad=50, fontsize=85)

plt.title("A Barplot comparing the testing data MSE \nof all four regressors", fontsize=100)
plt.show()
# obtain the ground truth and predictions of each regressor as lists
ground_truth = np.exp(price_scaler.inverse_transform(np.expand_dims(y_test, axis=1)))
sv_preds = np.exp(price_scaler.inverse_transform(np.expand_dims(sv_preds, axis=1)))
rf_preds = np.exp(price_scaler.inverse_transform(np.expand_dims(rf_preds, axis=1)))
kn_preds = np.exp(price_scaler.inverse_transform(np.expand_dims(kn_preds, axis=1)))
mlp_preds = np.exp(price_scaler.inverse_transform(mlp_preds))
# Set up the matplotlib figure
f, ax = plt.subplots(2, 2)
sns.scatterplot(x=ground_truth.flatten(), y=sv_preds.flatten(), ax=ax[0, 0], color='red', s=150)
sns.scatterplot(x=ground_truth.flatten(), y=rf_preds.flatten(), ax=ax[0, 1], color='blue', s=150)
sns.scatterplot(x=ground_truth.flatten(), y=kn_preds.flatten(), ax=ax[1, 0], color='green', s=150)
sns.scatterplot(x=ground_truth.flatten(), y=mlp_preds.flatten(), ax=ax[1, 1], color='purple', s=150)

# add subfigure titles
ax[0, 0].set_title('Support Vector Regressor', fontsize=60)
ax[0, 1].set_title('Random Forest Regressor', fontsize=60)
ax[1, 0].set_title('K-Nearest Neighbour Regressor', fontsize=60)
ax[1, 1].set_title('Deep Learning MLP Regressor', fontsize=60)

# generate annotations
annotations = []
for i, row in accuracy_df.iterrows():
    annotations.append('R2: ' + str(round(row[1], 3)) + '\nMSE: ' + str(np.format_float_scientific(np.asarray(row[2]), precision=3)))

# annotate the subfigures with the R-squared and MSE values
ax[0,0].annotate(annotations[0], xy=(0, max(sv_preds)*0.65), fontsize=40, color='dimgrey')
ax[0,1].annotate(annotations[1], xy=(0, max(rf_preds)*0.65), fontsize=40, color='dimgrey')
ax[1,0].annotate(annotations[2], xy=(0, max(kn_preds)*0.65), fontsize=40, color='dimgrey')
ax[1,1].annotate(annotations[3], xy=(0, max(mlp_preds)*0.65), fontsize=40, color='dimgrey')

# set all y-labels
plt.setp(ax[:, :], ylabel='Predicted \nvalues', xlabel='Observed values')

f.tight_layout(pad=2.0)

plt.show()
# a function to calculate residuals
def calculate_residuals(truth: list, preds: list):
    
    residuals = []
    for i in range(len(truth)):
        res = truth[i] - preds[i]
        residuals.append(res)
    
    return residuals

# calculate the residuals from the predictions of each regressor
sv_res = calculate_residuals(ground_truth, sv_preds)
rf_res = calculate_residuals(ground_truth, rf_preds)
kn_res = calculate_residuals(ground_truth, kn_preds)
mlp_res = calculate_residuals(ground_truth, mlp_preds)
# Set up the matplotlib figure
f, ax = plt.subplots(2, 2)
ax[0,0].hist(np.asarray(sv_res), color='red', edgecolor='black', linewidth=1.2)
ax[0,1].hist(np.asarray(rf_res), color='blue', edgecolor='black', linewidth=1.2)
ax[1,0].hist(np.asarray(kn_res), color='green', edgecolor='black', linewidth=1.2)
ax[1,1].hist(np.asarray(mlp_res), color='purple', edgecolor='black', linewidth=1.2)

# add subfigure titles
ax[0, 0].set_title('Support Vector Regressor', fontsize=60)
ax[0, 1].set_title('Random Forest Regressor', fontsize=60)
ax[1, 0].set_title('K-Nearest Neighbour Regressor', fontsize=60)
ax[1, 1].set_title('Deep Learning MLP Regressor', fontsize=60)

# annotate the subfigures with the median values
ax[0,0].annotate("Median: \n" + str(np.format_float_scientific(np.median(np.sort(np.asarray(sv_res))),
                                                             precision=3)), xy=(max(sv_res)*0.3, 40),
                 fontsize=45)

ax[0,1].annotate("Median: \n" + str(np.format_float_scientific(np.median(np.sort(np.asarray(rf_res))),
                                                             precision=3)), xy=(max(rf_res)*0.3, 40),
                 fontsize=45)

ax[1,0].annotate("Median: \n" + str(np.format_float_scientific(np.median(np.sort(np.asarray(kn_res))),
                                                             precision=3)), xy=(max(kn_res)*0.3, 40),
                 fontsize=45)

ax[1,1].annotate("Median: \n" + str(np.format_float_scientific(np.median(np.sort(np.asarray(mlp_res))),
                                                             precision=3)), xy=(max(mlp_res)*0.3, 40),
                 fontsize=45)


# set all x and y-labels
plt.setp(ax[:, :], ylabel='Frequency', xlabel='Residual values')

f.tight_layout(pad=1.4)

plt.show()