#targets["player_dismissed"].astype(str)

targets["player_dismissed"].fillna(0)

#targets["player_dismissed"].apply(lambda x: 0 if x == "NaN" else 1)
def distribution(data, transformed = False):

    """

    Visualization code for displaying skewed distributions of features

    """

    

    # Create figure

    fig = plt.figure(figsize = (11,5));



    # Skewed feature plotting

    for i, feature in enumerate(["over","ball"]):

        ax = fig.add_subplot(1, 2, i+1)

        ax.hist(data[feature], bins = 25, color = '#00A0A0')

        ax.set_title("'%s' Feature Distribution"%(feature), fontsize = 14)

        ax.set_xlabel("Value")

        ax.set_ylabel("Number of Records")

        ax.set_ylim((0, 2000))

        ax.set_yticks([0, 500, 1000, 1500, 2000])

        ax.set_yticklabels([0, 500, 1000, 1500, ">2000"])



    # Plot aesthetics

    if transformed:

        fig.suptitle("Log-transformed ", \

            fontsize = 16, y = 1.03)

    else:

        fig.suptitle("Skewed Distributions ", \

            fontsize = 16, y = 1.03)



    fig.tight_layout()

    fig.show()

    

distribution(features)    
targets.head()

from sklearn.preprocessing import MinMaxScaler



features["inning"] = features[features["inning"] <=2 ]

features["ball"] = features[features["ball"] <=7 ]

targets["total_runs"] = targets[targets["total_runs"] <=6 ]



scaler = MinMaxScaler()

f = ["ball","over"]

t = ["total_runs"]



#features[f] = scaler.fit_transform(features[f])

#targets[t] = scaler.fit_transform(targets[t])
from sklearn.preprocessing import LabelEncoder

lb = LabelEncoder()

lb.fit(features[["inning","ball","over"]])

features[["inning","ball","over"]] = lb.transform(features[["inning","ball","over"]])
# Import train_test_split

from sklearn.cross_validation import train_test_split



# Split the 'features' and 'income' data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size = 0.2, random_state = 0)



# Show the results of the split

print ("Training set has {} samples.".format(X_train.shape))

print ("Testing set has {} samples.".format(y_train.shape))



#from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

#from sklearn import tree

from sklearn.ensemble import RandomForestClassifier



#clf = GaussianNB()

clf = RandomForestClassifier(random_state=0)

#clf_B = KNeighborsClassifier()

#clf_C = tree.DecisionTreeClassifier(random_state = 0)



clf.fit(X_train, y_train)

predictions = clf.predict(X_test)

print (accuracy_score(y_test, predictions, normalize=True))
# TODO: Import 'r2_score'

from sklearn.metrics import r2_score



def performance_metric(y_true, y_predict):

    """ Calculates and returns the performance score between 

        true and predicted values based on the metric chosen. """

    

    # TODO: Calculate the performance score between 'y_true' and 'y_predict'

    score = r2_score(y_true,y_predict)

    

    # Return the score

    return score
print (performance_metric(y_test, predictions))
# TODO: Import 'make_scorer', 'DecisionTreeRegressor', and 'GridSearchCV'

from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import make_scorer

from sklearn.grid_search import GridSearchCV

from sklearn.cross_validation import ShuffleSplit





def fit_model(X, y):

    """ Performs grid search over the 'max_depth' parameter for a 

        decision tree regressor trained on the input data [X, y]. """

    

    # Create cross-validation sets from the training data

    cv_sets = ShuffleSplit(X.shape[0], n_iter = 10, test_size = 0.20, random_state = 0)



    # TODO: Create a decision tree regressor object

    regressor = DecisionTreeRegressor()



    # TODO: Create a dictionary for the parameter 'max_depth' with a range from 1 to 10

    params = {'max_depth':list(range(1,11))}



    # TODO: Transform 'performance_metric' into a scoring function using 'make_scorer' 

    scoring_fnc = make_scorer(performance_metric)



    # TODO: Create the grid search object

    grid = GridSearchCV(regressor,params,scoring=scoring_fnc,cv=cv_sets)



    # Fit the grid search object to the data to compute the optimal model

    grid = grid.fit(X, y)



    # Return the optimal model after fitting the data

    return grid.best_estimator_
# Fit the training data to the model using grid search

reg = fit_model(X_train, y_train)



# Produce the value for 'max_depth'

#print ("Parameter 'max_depth' is {} for the optimal model.".format(reg.get_params()['max_depth']))
# Produce a matrix for client data

#client_data = [[1,"Kolkata Knight Riders", "Royal Challengers Bangalore",1,5,"BB McCullum",

                #"P Kumar"]]  # Client 3



# Show predictions

#for i, price in enumerate(reg.predict(client_data)):

    #print ("Predicted selling price for Client {}'s home: ${:,.2f}".format(i+1, price))

#print (accuracy_score(y_test, reg.predict(X_test), normalize=True))

from sklearn.metrics import r2_score

print (r2_score(y_test,reg.predict(X_test)))
