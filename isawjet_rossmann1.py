import pandas as pd
train_data = pd.read_csv('C://Users//tteja//Desktop//data//Kaggle//Rossmann Store Sales//train.csv')

from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation

predictors = ["Store","DayOfWeek","Date","Customers","Open","Promo","StateHoliday","SchoolHoliday"]

alg = RandomForestClassifier(
    random_state=1,
    n_estimators=150,
    min_samples_split=4,
    min_samples_leaf=2
)

scores = cross_validation.cross_val_score(
    alg,
    train[predictors],
    train["Sales"],
    cv=3
)

print(scores.mean())
