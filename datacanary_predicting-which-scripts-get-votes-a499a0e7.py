import pandas as pd



import sqlite3







con = sqlite3.connect('../input/database.sqlite')







scripts = pd.read_sql_query("""



SELECT s.Id,



       cv.Title,



       COUNT(DISTINCT vo.Id) NumVotes,



       COUNT(DISTINCT CASE WHEN vo.UserId!=s.AuthorUserId THEN vo.Id ELSE NULL END) NumNonSelfVotes,



       CASE WHEN COUNT(DISTINCT CASE WHEN vo.UserId!=s.AuthorUserId THEN vo.Id ELSE NULL END)>0 THEN 1 ELSE 0 END HasNonSelfVotes,



       COUNT(DISTINCT v.Id) NumVersions,



       SUM(CASE WHEN r.WorkerStatus=2 THEN 1 ELSE 0 END) NumSuccessfulRuns,



       SUM(CASE WHEN r.WorkerStatus=3 THEN 1 ELSE 0 END) NumErroredRuns,



       SUM(CASE WHEN v.IsChange=1 THEN 1 ELSE 0 END) NumChangedVersions,



       SUM(v.LinesInsertedFromPrevious-v.LinesDeletedFromPrevious) Lines,



       SUM(v.LinesInsertedFromPrevious+v.LinesChangedFromPrevious) LinesAddedOrChanged,



       l.Name



FROM Scripts s



INNER JOIN ScriptVersions v ON v.ScriptId=s.Id



INNER JOIN ScriptVersions cv ON s.CurrentScriptVersionId=cv.Id



INNER JOIN ScriptRuns r ON r.ScriptVersionId=v.Id



INNER JOIN ScriptLanguages l ON v.ScriptLanguageId=l.Id



LEFT OUTER JOIN ScriptVotes vo ON vo.ScriptVersionId=v.Id



WHERE r.WorkerStatus != 4



  AND r.WorkerStatus != 5



GROUP BY s.Id,



         cv.Title,



         cv.Id,



         l.Name



ORDER BY cv.Id DESC



""", con)







scripts
from sklearn.pipeline import Pipeline, FeatureUnion



from sklearn.cross_validation import train_test_split



from sklearn.ensemble import RandomForestClassifier







class RawColumnExtractor:



    def __init__(self, column):



        self.column=column







    def fit(self, *_):



        return self



    



    def transform(self, data):



        return data[[self.column]]







features = FeatureUnion([("NumSuccessfulRuns",  RawColumnExtractor("NumSuccessfulRuns")),



                         ("NumChangedVersions", RawColumnExtractor("NumChangedVersions"))



                        ])







pipeline = Pipeline([('feature_union',  features),



                     ('predictor',      RandomForestClassifier())



                    ])







train = scripts



target_name = "HasNonSelfVotes"







x_train, x_test, y_train, y_test = train_test_split(train, train[target_name], test_size=0.4, random_state=0)







pipeline.fit(x_train, y_train)



score = pipeline.score(x_test, y_test)



print("Score %f" % score)

pd.read_sql_query("""



SELECT *



FROM ScriptLanguages



LIMIT 100



""", con)