from sklearn import linear_model 

model = linear_model.Ridge(alpha=0.1) 

model.fit([[-1], [0], [1]], [-1,0,1]) 

print(model.coef_) 

print(model.intercept_) 

model.predict([[3]]) 