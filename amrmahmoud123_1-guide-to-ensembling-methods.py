model1 = tree.DecisionTreeClassifier()

model2 = KNeighborsClassifier()

model3= LogisticRegression()



model1.fit(x_train,y_train)

model2.fit(x_train,y_train)

model3.fit(x_train,y_train)



pred1=model1.predict(x_test)

pred2=model2.predict(x_test)

pred3=model3.predict(x_test)



final_pred = np.array([])

for i in range(0,len(x_test)):

    final_pred = np.append(final_pred, mode([pred1[i], pred2[i], pred3[i]]))
from sklearn.ensemble import VotingClassifier

model1 = LogisticRegression(random_state=1)

model2 = tree.DecisionTreeClassifier(random_state=1)

model = VotingClassifier(estimators=[('lr', model1), ('dt', model2)], voting='hard')

model.fit(x_train,y_train)

model.score(x_test,y_test)
model1 = tree.DecisionTreeClassifier()

model2 = KNeighborsClassifier()

model3= LogisticRegression()



model1.fit(x_train,y_train)

model2.fit(x_train,y_train)

model3.fit(x_train,y_train)



pred1=model1.predict_proba(x_test)

pred2=model2.predict_proba(x_test)

pred3=model3.predict_proba(x_test)



finalpred=(pred1+pred2+pred3)/3
model1 = tree.DecisionTreeClassifier()

model2 = KNeighborsClassifier()

model3= LogisticRegression()



model1.fit(x_train,y_train)

model2.fit(x_train,y_train)

model3.fit(x_train,y_train)



pred1=model1.predict_proba(x_test)

pred2=model2.predict_proba(x_test)

pred3=model3.predict_proba(x_test)



finalpred=(pred1*0.3+pred2*0.3+pred3*0.4)
final_dt = DecisionTreeClassifier(max_leaf_nodes=10, max_depth=5)                   

final_bc = BaggingClassifier(base_estimator=final_dt, n_estimators=40, random_state=1, oob_score=True)



final_bc.fit(X_train, train_y)

final_preds = final_bc.predict(X_test)





acc_oob = final_bc.oob_score_

print(acc_oob)  

from sklearn.ensemble import RandomForestClassifier

model= RandomForestClassifier(random_state=1)

model.fit(x_train, y_train)

model.score(x_test,y_test)
from sklearn.ensemble import RandomForestRegressor

model= RandomForestRegressor()

model.fit(x_train, y_train)

model.score(x_test,y_test)
from sklearn.ensemble import AdaBoostClassifier

model = AdaBoostClassifier(random_state=1)

model.fit(x_train, y_train)

model.score(x_test,y_test)





#Sample code for regression problem:



from sklearn.ensemble import AdaBoostRegressor

model = AdaBoostRegressor()

model.fit(x_train, y_train)

model.score(x_test,y_test)