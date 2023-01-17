from sklearn.linear_model import ElasticNet

model = ElasticNet(alpha=0.1)

model.fit([[-1], [0], [1]], [-1,0,1])

print(model.coef_) 

print(model.intercept_) 

model.predict([[3]]) 
#Add a second feature



from sklearn.linear_model import ElasticNet 

model = ElasticNet(alpha=0.1) 

X_train = [[-1,-1],[0,0],[1,1]]  

Y_train = [-1,0,1] 

model.fit(X_train,Y_train)

print(model.intercept_) 

print(model.coef_) 

model.predict([[3,-3]]) 