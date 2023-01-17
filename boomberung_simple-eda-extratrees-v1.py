# True to spend extra time displaying graphs, False for speedy results
show_plots = True
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white")

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings("ignore")

%matplotlib inline
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.info()
train.describe()
train.head()
target_classes = range(1,8)
target_class_names = ['Spruce/Fir', 'Lodgepole Pine', 'Ponderosa Pine', \
                      'Cottonwood/Willow', 'Aspen', 'Douglas-fir', 'Krummholz']

numerical_features = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', \
                    'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways', \
                    'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points']

categorical_features = [ 'Wilderness_Area', 'Soil_Type' ]
# extract target from data
y = train['Cover_Type']
train = train.drop('Cover_Type', axis=1)
# plot target var
plt.hist(y, bins='auto')
plt.title('Cover_Type')
plt.xlabel('Class')
plt.ylabel('# Instances')
plt.show()
if show_plots:
    for feature_name in numerical_features:
        plt.figure()
        sns.distplot(train[feature_name], label='train')
        sns.distplot(test[feature_name], label='test')
        plt.legend()
        plt.show()
if show_plots:
    # categorical distributions btw train and test set
    train_wilderness_categorical = train['Wilderness_Area1'].copy().rename('Wilderness_Area')
    train_wilderness_categorical[train['Wilderness_Area2'] == 1] = 2
    train_wilderness_categorical[train['Wilderness_Area3'] == 1] = 3
    train_wilderness_categorical[train['Wilderness_Area4'] == 1] = 4

    test_wilderness_categorical = test['Wilderness_Area1'].copy().rename('Wilderness_Area')
    test_wilderness_categorical[test['Wilderness_Area2'] == 1] = 2
    test_wilderness_categorical[test['Wilderness_Area3'] == 1] = 3
    test_wilderness_categorical[test['Wilderness_Area4'] == 1] = 4

    plt.figure()
    sns.countplot(train_wilderness_categorical, label='train')
    plt.title('Wilderness_Area in Train')

    plt.figure()
    sns.countplot(test_wilderness_categorical, label='test')
    plt.title('Wilderness_Area in Test')

    plt.show()
soil_classes = range(1,41)

train_soiltype_categorical = train['Soil_Type1'].copy().rename('Soil_Type')
for cl in soil_classes:
    train_soiltype_categorical[train['Soil_Type'+str(cl)] == 1] = cl

test_soiltype_categorical = test['Soil_Type1'].copy().rename('Soil_Type')
for cl in soil_classes:
    test_soiltype_categorical[test['Soil_Type'+str(cl)] == 1] = cl

plt.figure(figsize=(10, 5))
sns.countplot(train_soiltype_categorical, label='train')
plt.title('Soil_Type in Train')

plt.figure(figsize=(10, 5))
sns.countplot(test_soiltype_categorical, label='test')
plt.title('Soil_Type in Test')

plt.show()
pca = PCA(n_components=3)
train_pca = pca.fit_transform(train)
print('Representation of dataset in 3 dimensions:\n')
print(train_pca)
if show_plots:
    # graph pca in interactive 3d chart
    # props to Roman Kovalenko's "Data distribution & 3D Scatter Plots" kernel for showing me where to find a good 3d graphing lib

    colors = ['red', 'blue', 'green', 'black', 'purple', 'orange', 'gray']
    # feel free to change the colors up - unfortunately there's usually a tradeoff between aesthetics and readability
    # colors = ['#f45f42', '#f49241', '#db6a0d', '#dba00d', '#ead40e', '#ffb163', '#ea480e']

    traces = []

    # iterate over classes and add each set of points to traces list
    for cl in target_classes:

        # get all 3-d pca vectors that match the current class
        class_pca = train_pca[y[y == cl].index.values]

        class_pca_x = [ pt[0] for pt in class_pca]
        class_pca_y = [ pt[1] for pt in class_pca]
        class_pca_z = [ pt[2] for pt in class_pca]

        trace = go.Scatter3d(
            x=class_pca_x,
            y=class_pca_y,
            z=class_pca_z,
            mode='markers',
            marker=dict(
                color=colors[cl-1],
                size=3
            ),
            name=target_class_names[cl-1]
        )

        traces.append(trace)

    layout = go.Layout(
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0
        )
    )
    fig = go.Figure(data=traces, layout=layout)
    py.iplot(fig, filename='simple-3d-scatter')
# drop uninformative features
train = train.drop('Id', axis=1)
# write a function to transform the train and test sets
# we'll also append an underscore "_" to our engineered feature names to help differentiate them
def add_features(data):
    data['Euclidean_Distance_To_Hydrology_'] = (data['Horizontal_Distance_To_Hydrology']**2 + data['Vertical_Distance_To_Hydrology']**2)**0.5
    data['Mean_Distance_To_Amenities_'] = (data['Horizontal_Distance_To_Fire_Points'] + data['Horizontal_Distance_To_Hydrology'] + data['Horizontal_Distance_To_Roadways']) / 3.0
    data['Elevation_Minus_Vertical_Distance_To_Hydrology_'] = data['Elevation'] - data['Vertical_Distance_To_Hydrology']
    return data

train = add_features(train)
test = add_features(test)
# # convert aspect angle in degrees to cos + sin
# train['AspectCos'] = train['Aspect']
# train['AspectSin'] = train['Aspect']

# train['AspectCos'] = train['AspectCos'].apply(lambda x: np.cos(np.deg2rad(x)))
# train['AspectSin'] = train['AspectSin'].apply(lambda x: np.sin(np.deg2rad(x)))

# train = train.drop(['Aspect'], axis=1)
if show_plots:
    # plot each feature (y axis) with target (x axis)
    plt.figure(figsize=(30, 190))

    # iterate through feature names and assign to pyplot subplot
    for i,feature_name in enumerate(train.columns.values):
        plt.subplot(19,3,i+1)
        sns.violinplot(y, train[feature_name])
        plt.title(feature_name, fontsize=30)

    plt.show()
if show_plots:

    # Compute the correlation matrix
    corr = train.corr()

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(110, 90))

    # Generate a custom diverging colormap, use the line below to customize your color options
    # sns.choose_diverging_palette()
    cmap = sns.diverging_palette(8,132,99,50,50,9, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns_heatmap = sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0)

    # since the heatmap is very large, use following line to save to png for close examination
    # sns_heatmap.get_figure().savefig("corr_heatmap.png")
# split data into train and test sets, using constant random state to better quantify our changes
X_train, X_test, y_train, y_test = train_test_split(train, y, test_size=0.33, random_state=1)
# train model
model = ExtraTreesClassifier(n_estimators=500)
model.fit(X_train, y_train)
# plot feature importance
plt.figure(figsize=(20,20))
plt.barh(X_train.columns.values, model.feature_importances_)
plt.title('Feature Importance')
plt.ylabel('Feature Name')
plt.xlabel('Gini Value')
plt.show()
# make predictions on the cross validation set
y_pred = model.predict(X_test)
n_correct = (y_pred == y_test).sum()
n_total = (y_pred == y_test).count()
print('Accuracy:', n_correct/n_total)
# table with data points, truth, and pred
errors = X_test.copy()
errors['truth'] = y_test
errors['pred'] = y_pred
errors = errors[errors['truth'] != errors['pred']]
print(errors.shape[0], 'errors over',y_pred.shape[0],'predictions')
errors.head()
errors.describe()
errors.describe() - train.describe()
# x: classes y: # errors
error_truths = []
for cl in target_classes:
    error_count = errors[errors['truth'] == cl]['truth'].count()
    error_truths.append(error_count)
    
plt.bar(target_classes, error_truths)
plt.title('Errors by truth class')
plt.xlabel('True Class')
plt.ylabel('# Errors')
plt.show()
# x: classes y: # errors
error_preds = []
for cl in target_classes:
    error_count = errors[errors['pred'] == cl]['pred'].count()
    error_preds.append(error_count)
    
plt.bar(target_classes, error_preds)
plt.title('Errors by predicted class')
plt.xlabel('Predicted Class')
plt.ylabel('# Errors')
plt.show()
cf_matrix = confusion_matrix(errors['truth'], errors['pred'])

cfm_df = pd.DataFrame(cf_matrix, index = [str(cl)+'t' for cl in target_classes],
                  columns = [str(cl)+'p' for cl in target_classes])

ax = plt.axes()
sns.heatmap(cfm_df, annot=True, fmt='g', ax=ax)
ax.set_title('Error Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()
prediction_classes = pd.Series(model.predict(test.drop('Id', axis=1))).rename('Cover_Type')
predictions = pd.concat([test['Id'], prediction_classes], axis=1).reset_index().drop('index', axis=1)
predictions.to_csv('submission.csv', index=False)
predictions.head()