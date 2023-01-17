import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
#import data into dataframe
rec = pd.read_csv('./BeerRecipes/recipeData.csv', encoding = "ISO-8859-1")
#there are 22 columns and 73K rows
rec.shape
#quickly view which columns contain nulls, these are not easily imputable, and there is lots of missind data, will drop
rec.isnull().sum()
#drop columns with lots of missing data
rec.drop(columns = ['MashThickness', 'PitchRate', 'PrimaryTemp', 'PrimingMethod', 'PrimingAmount', 'BoilGravity'], inplace = True)
#drop columns that won't be used in prediction, non-recipe specific columns
rec.drop(columns = ['Name', 'BeerID', 'Size(L)', 'BoilSize', 'BrewMethod', 'URL', 'SugarScale'], inplace = True)
#drop rows that do not have a style, this is our y(what we will predict)
rec.drop(rec[rec['Style'].isnull()].index, inplace =  True)

#one hot encoding styles
rec['IPA'] = rec['Style'].apply(lambda x: 1 if x.find('IPA') > -1 else 0)
rec['Porter'] = rec['Style'].apply(lambda x: 1 if x.find('Porter') > -1 else 0)
rec['Stout'] = rec['Style'].apply(lambda x: 1 if x.find('Stout') > -1 else 0)
rec['Ale'] = rec['Style'].apply(lambda x: 1 if x.find('Ale') > -1 else 0)
rec['Lager'] = rec['Style'].apply(lambda x: 1 if x.find('Lager') > -1 else (1 if x.find('Pils') > -1 else 0))
rec['Witbier'] = rec['Style'].apply(lambda x: 1 if x.find('Witbier') > -1 else (1 if x.find('wheat') > -1 else 0))
rec['Saison'] = rec['Style'].apply(lambda x: 1 if x.find('Saison') > -1 else 0)

#defined styles present in dataset
rec[['IPA', 'Porter', 'Stout', 'Ale', 'Lager', 'Witbier', 'Saison']].sum().sum()
#drop styles that don't fit into pre-defined categories
rec.drop(rec[(rec['IPA'] != 1) & (rec['Porter'] != 1) & (rec['Stout'] != 1) & (rec['Ale'] != 1) & (rec['Lager'] != 1) \
                      & (rec['Witbier'] != 1) & (rec['Saison'] != 1)].index, inplace = True)
#reset the index 
rec.reset_index(drop = True, inplace = True)

#setting column of ids for newly declared styles, reverse one-hot-encoded
rec['New_StyleID'] = 0

for i in range(rec.shape[0]):
    if rec.loc[i, 'IPA'] == 1:
        rec.loc[i, 'New_StyleID'] = 'IPA'
    elif rec.loc[i, 'Porter'] == 1:
        rec.loc[i, 'New_StyleID'] = 'Porter'
    elif rec.loc[i, 'Stout'] == 1:
        rec.loc[i, 'New_StyleID'] = 'Stout'
    elif rec.loc[i, 'Ale'] == 1:
        rec.loc[i, 'New_StyleID'] = 'Ale'
    elif rec.loc[i, 'Lager'] == 1:
        rec.loc[i, 'New_StyleID'] = 'Lager'
    elif rec.loc[i, 'Witbier'] == 1:
        rec.loc[i, 'New_StyleID'] = 'Witbier'
    elif rec.loc[i, 'Saison'] == 1:
        rec.loc[i, 'New_StyleID'] = 'Saison'
rec.head()
#saving count of top style, American IPA, to determine baseline accuracy
top_style = rec.groupby(by = 'Style')[['Style']].count().sort_values(by = 'Style', ascending = False).rename(columns = {'Style':'Style Count'}).reset_index().loc[0,'Style Count']

#showing counts of granular styles in dataset
rec.groupby(by = 'Style')[['Style']].count().sort_values(by = 'Style', ascending = False).rename(columns = {'Style':'Style Count'}).reset_index().tail()
#dataframe showing totals of styles
style_agg = pd.DataFrame(rec[['IPA', 'Porter', 'Stout', 'Ale', 'Lager', 'Witbier', 'Saison']].sum(), columns = ['Style Count']).reset_index().rename(columns = {'index':'Style'}).sort_values(by = 'Style Count', ascending = False)
style_agg['percent'] = style_agg['Style Count'].apply(lambda x: x/style_agg['Style Count'].sum()*100)
style_agg
#plotting counts of styles
style_agg.plot(x = 'Style', y = 'percent', kind = 'bar', legend = False)
plt.title('Percent of Total of Beer Styles')
plt.xlabel('Recipe Style')
plt.ylabel('Recipe Percent of Total')
plt.xticks(rotation = 0);
#histogram of color
rec.hist(column = 'Color', grid = False, bins = 15)
plt.xlim(-3, 65)
plt.title('Standard Reference Method (SRM) Color Histogram')
plt.xlabel('Recipe SRMs')
plt.ylabel('Count of Colors');
#boxplots of Color of beer styles
sns.factorplot(kind = 'box', 
               y = 'Color',
               x = 'New_StyleID',
               data = rec,
               size = 5,
               aspect = 1.5)
plt.title('Boxplots of Color for Beer Styles')
plt.xlabel('Beer Style')
plt.ylabel('Color (SRM)')
plt.ylim([0,100]);
#boxplot of ABV of beer styles
sns.factorplot(kind = 'box', 
               y = 'ABV',
               x = 'New_StyleID',
               data = rec,
               size = 5,
               aspect = 1.5)
plt.title('Boxplots of ABV for Beer Styles')
plt.xlabel('Beer Style')
plt.ylabel('Alcohol by Volumne (ABV)')
plt.ylim([0,15]);
#boxplot of IBUs of beer styles
sns.factorplot(kind = 'box', 
               y = 'IBU',
               x = 'New_StyleID',
               data = rec,
               size = 5,
               aspect = 1.5)
plt.title('Boxplots of IBUs for Beer Styles')
plt.xlabel('Beer Style')
plt.ylabel('International Bittering Units (IBUs)')
plt.ylim([0,200]);
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
#declare X and y
X = rec[['OG', 'FG', 'ABV', 'IBU', 'Color', 'BoilTime', 'Efficiency']]
y = rec[['IPA', 'Porter', 'Stout', 'Ale', 'Lager', 'Witbier', 'Saison']]
#train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

#set as numpy array
y_train = y_train.as_matrix()
y_test = y_test.as_matrix()
#standardizing data to normalize with a mean of 0 and a stdDev of 1
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.model_selection import GridSearchCV

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
def CART_model_picker(model_list, X_train, y_train, X_test, y_test):
    '''Returns cross-val score and score for all input CART models, returns in sorted order of best performing'''
    results = []
    
    for (k,v) in model_list.items():
        scores = cross_val_score(v, X_train, y_train, cv = 5)
        v.fit(X_train, y_train)

        results.append((k, scores.mean(), v.score(X_test, y_test)))
    
    results.sort(key = lambda x: x[2], reverse = True)
    results.insert(0, ('Model', 'Cross_val_score mean', 'Model Score R^2'))
    
    return results
#figure out which CART model performs the best
models = {'Decision Tree': DecisionTreeClassifier(),
          'Extra Trees': ExtraTreesClassifier(),
          'Random Forest':RandomForestClassifier()
         }

dt_results = CART_model_picker(models, X_train, y_train, X_test, y_test)
dt_results
#plotting results from model picker
cart_res = pd.DataFrame(dt_results[1:], columns = ['Model', 'Cross_val_mean', 'R2']).sort_values(by = 'R2', ascending = False)
cart_res.plot(x = 'Model', y = ['Cross_val_mean', 'R2'], kind = 'bar')
plt.title('Results of CART Models')
plt.xlabel('CART Model')
plt.ylabel('Model Metrics')
plt.xticks(rotation=0);
#dataframe of feature importances
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
dtc_feats = pd.DataFrame(list(zip(['OG', 'FG', 'ABV', 'IBU', 'Color', 'BoilTime', 'Efficiency'], dtc.feature_importances_)), \
                         columns = ['Feature', 'Importance']).sort_values(by = 'Importance', ascending = False)

#plotting feature importances for top decision tree classifier
dtc_feats.plot(x = 'Feature', y = 'Importance', kind = 'bar', legend = False)
plt.title('Decision Tree Classifier Feature Importances')
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.xticks(rotation=0);
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import regularizers
#create model
model = Sequential()
#network topology
input_units = X_train.shape[1]
hidden_units = 4

#input layer
model.add(Dense(hidden_units, 
                input_dim = input_units, 
                activation = 'relu',
                kernel_regularizer=regularizers.l2(0.0001)))
#hidden layer
model.add(Dense(50, activation = 'relu',))
#hidden layer
model.add(Dense(50, activation = 'relu',))


#output layer
model.add(Dense(7, activation = 'softmax'))
#compile the model
model.compile(loss = 'categorical_crossentropy',
              optimizer = 'rmsprop',
              metrics = ['accuracy'])
#train the model
history = model.fit(X_train, y_train,
                   validation_data = (X_test, y_test),
                   epochs = 30, 
                   batch_size = None, 
                   verbose = 1)
#plotting accuracy over epochs
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Neural Network Accuracy over Epochs')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['Test', 'Train']);
train_loss = history.history['loss']
test_loss = history.history['val_loss']

plt.plot(train_loss, label = 'Train loss')
plt.plot(test_loss, label = 'Test loss')
plt.legend();
plt.title('Neural Network Test and Train Loss Over Epochs')
plt.ylabel('L2 Loss')
plt.xlabel('Epochs')
plt.legend(['Test', 'Train']);

