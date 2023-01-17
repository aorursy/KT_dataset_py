import pandas as pd



cats = pd.read_csv("../input/cookie_cats.csv")

display(len(cats))

display(cats.head())
cats[['userid', 'version']].groupby('version').count()
%matplotlib inline

 

plot_cats = cats[['userid', 'sum_gamerounds']].groupby('sum_gamerounds').count()



ax = plot_cats.head(100).plot()

ax.set_xlabel("number of games played")

ax.set_ylabel("number of users")
cats['retention_1'].mean()*100
cats[['retention_1', 'version']].groupby('version').mean() * 100
boot_1d = []

for i in range(500):

    boot_mean = cats[['retention_1', 'version']].sample(frac = 1, replace = True).groupby('version').mean() * 100

    boot_1d.append(boot_mean)

    

boot_1d = pd.DataFrame([[boot_1d[i].values[0,0], boot_1d[i].values[1,0]] for i in range(500)], columns=['gate_30','gate_40'])

    

# A Kernel Density Estimate plot of the bootstrap distributions

boot_1d.plot.kde(figsize = (11,5))
boot_1d['diff'] = (boot_1d['gate_30'] - boot_1d['gate_40']) / boot_1d['gate_40'] * 100



# Ploting the bootstrap % difference

ax = boot_1d['diff'].plot.kde(figsize = (11, 5))

ax.set_xlabel("percentage difference in 1-day retention")
prob = (boot_1d['diff'] > 0.0).mean()

print(prob * 100)
cats[['retention_7', 'version']].groupby('version').mean() * 100
boot_7d = []

for i in range(500):

    boot_mean = cats[['retention_7', 'version']].sample(frac = 1, replace = True).groupby('version').mean() * 100

    boot_7d.append(boot_mean)

    

boot_7d = pd.DataFrame([[boot_7d[i].values[0,0], boot_7d[i].values[1,0]] for i in range(500)], columns=['gate_30','gate_40'])



boot_7d['diff'] = (boot_7d['gate_30'] - boot_7d['gate_40']) / boot_7d['gate_40'] * 100



ax = boot_7d['diff'].plot.kde(figsize = (11, 5))

ax.set_xlabel("% difference in means")



prob = (boot_7d['diff'] > 0.0).mean()



print(prob * 100)