import pandas as pd

import numpy as np

import seaborn as sns

from sklearn.linear_model import LinearRegression,Ridge, Lasso

from sklearn.model_selection import cross_val_score

import matplotlib

%pylab inline

pd.options.display.max_columns = 300
train = pd.read_csv("../input/train.csv")

target = train["SalePrice"]

train = train.drop("SalePrice",1) # take out the target variable

test = pd.read_csv("../input/test.csv")

combi = pd.concat((train,test)) # this is the combined data frame without the target variable
print(shape(train))

print(shape(test))

print(shape(combi))
figure(figsize(8,4))

subplot(1,2,1)

hist(target*1e-6,20);

xlabel("Sale Price in Mio Dollar")

subplot(1,2,2)

hist(log10(target),20);

xlabel("log10(Sale Price in Dollar)")
target = log10(target)
combi.head(10)
# create new features from categorical data:

combi = pd.get_dummies(combi)

# and fill missing entries with the column mean:

combi = combi.fillna(combi.mean())



# create the new train and test arrays:

train = combi[:train.shape[0]]

test = combi[train.shape[0]:]
combi.head(10)
model = LinearRegression()

score = mean(sqrt(-cross_val_score(model, train, target,scoring="neg_mean_squared_error", cv = 5)))

print("linear regression score: ", score)
cv = 5 #number of folds in cross-validation



alphas = np.logspace(-5,2,20)

scores = np.zeros((len(alphas),cv))

scores_mu = np.zeros(len(alphas))

scores_sigma = np.zeros(len(alphas))



for i in range(0,len(alphas)):

    model = Ridge(alpha=alphas[i])

    scores[i,:] = sqrt(-cross_val_score(model, train, target,scoring="neg_mean_squared_error", cv = cv))

    scores_mu[i] = mean(scores[i,:])

    scores_sigma[i] = std(scores[i,:])



figure(figsize(8,4))   

#for i in range(0,cv):

#    plot(alphas,scores[:,i], 'b--', alpha=0.5)

plot(alphas,scores_mu,'c-',lw=3, alpha=0.5, label = "Ridge")

fill_between(alphas,np.array(scores_mu)-np.array(scores_sigma),

             np.array(scores_mu)+np.array(scores_sigma),color="c",alpha=0.5)



print("best score in Ridge: ",min(scores_mu))



for i in range(0,len(alphas)):

    model = Lasso(alpha=alphas[i])

    scores[i,:] = sqrt(-cross_val_score(model, train, target,scoring="neg_mean_squared_error", cv = cv))

    scores_mu[i] = mean(scores[i,:])

    scores_sigma[i] = std(scores[i,:])



plot(alphas,scores_mu,'g-',lw=3, alpha=0.5, label="Lasso")

fill_between(alphas,np.array(scores_mu)-np.array(scores_sigma),

             np.array(scores_mu)+np.array(scores_sigma),color="g",alpha=0.5)



xscale("log")

plt.xlabel("alpha", size=20)

plt.ylabel("rmse", size=20)

legend(loc=2)



print("best score in Lasso: ",min(scores_mu))
model = Lasso(alpha=1e-4)

model.fit(train,target)

preds = model.predict(test)

preds = 10**preds
solution = pd.DataFrame({"id":test.Id, "SalePrice":preds})

solution.to_csv("submit.csv", index = False)