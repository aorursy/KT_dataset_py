from pandas import read_csv

data = read_csv('../input/glass.csv')

X = data.drop(["Type"],1)

y = data["Type"]
from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier()
def get_row(model, feature_data, y):

    from time import time

    start = time()

    from sklearn.model_selection import cross_val_score

    scores = cross_val_score(model,feature_data,y,cv=3, n_jobs=1)

    finish = time() - start

    time_finished = "%d minutes%2d seconds" % (int(finish / 60), finish % 60) 

    unit = {}

    unit["Number of Features"] = len(feature_data.columns)

    unit["Average Score"] = scores.mean()

    unit["Standard Deviation"] = scores.std()

    unit["Processing Time"] = time_finished

    return unit



from pandas import DataFrame

from itertools import combinations



def feature_set_score_dataframe(model, X, y):

    

    features = X.columns

    current = {}

    

    from itertools import combinations

    

    for i in range(2,len(features)):

        

        for comb in combinations(features,i):

            name = str(list(comb))

            current[name] = get_row(model, X[list(comb)],y)

        

    output = DataFrame(current).T

    

    return output
result = feature_set_score_dataframe(model, X, y)
from IPython.display import display

display(result.sort_values("Average Score", ascending=False).head(10))