# Packages
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import seaborn as sb
import pystan
from scipy.stats import pearsonr, zscore

# Plot settings
%matplotlib inline
%config InlineBackend.figure_format = 'svg'
sb.set()

# Settings for Stan
Niters = 1000 #number of MCMC iterations
Nchains = 4   #number of MCMC chains
# Create some dummy data with known correlation
X = np.zeros((20,2))
X[:,0] = np.linspace(0, 20, 20)
X[:,1] = X[:,0] + np.random.randn(20)

# Plot dummy data
plt.figure()
plt.plot(X[:,0], X[:,1], '.')
plt.xlabel(r'$X_1$')
plt.ylabel(r'$X_2$')
plt.title('Correlated variables')
plt.show()
# Compute Pearson correlation coefficient
rho, pval = pearsonr(X[:,0], X[:,1])
print('Correlation coefficient: %0.3g ( p = %0.3g )' % (rho, pval))
#STAN code string for a basic pearson correlation
PearsonCorrelation = """
data {
    int<lower=0> N; //number of datapoints
    vector[2] X[N]; //datapoints
}

parameters {
    vector[2] mu;               //mean
    vector<lower=0>[2] sig;     //std dev of each variable
    real<lower=-1,upper=1> rho; //Pearson's rho
}

transformed parameters {
    // Compute the covariance matrix from rho and sigmas
    cov_matrix[2] C;
    C[1,1] = sig[1] * sig[1];
    C[1,2] = rho * sig[1] * sig[2];
    C[2,1] = rho * sig[1] * sig[2];
    C[2,2] = sig[2] * sig[2];
}

model {
    // Model our data as being drawn from multivariate normal 
    // distribution with mean mu and covariance matrix C
    X ~ multi_normal(mu, C);
}
"""
# Compile the Stan model
model_pc = pystan.StanModel(model_code=PearsonCorrelation)
# Data for Stan
data = {'N' : X.shape[0], #number of datapoints
        'X' : X}          #the data

# Fit the model
fit = model_pc.sampling(data=data, iter=Niters, chains=Nchains, n_jobs=Nchains)
# Print the results of the fit
print(fit)
# Get the MCMC samples (draws from the posterior distribution)
samples = fit.extract()
# Plot the posterior distribution for the correlation coefficient
plt.figure()
sb.distplot(samples['rho'])
plt.xlabel('Pearson Correlation Coefficient')
plt.ylabel('Posterior Probability')
plt.show()
# Plot the posterior joint distribution for the means of the gaussian
plt.figure()
sb.kdeplot(samples['mu'][:,0], samples['mu'][:,1], 
           shade=True, shade_lowest=False, cbar=True)
plt.xlabel(r'$X_1$ Mean')
plt.ylabel(r'$X_2$ Mean')
plt.title('Posterior Joint Distribution of the Means')
plt.show()
# Plot the posterior joint distribution for the standard deviations of the gaussian
plt.figure()
sb.kdeplot(samples['sig'][:,0], samples['sig'][:,1], 
           n_levels=5, cbar=True)
plt.xlabel(r'$X_1$ Std Dev')
plt.ylabel(r'$X_2$ Std Dev')
plt.title('Posterior Joint Distribution of the Variances')
plt.show()
# Create some dummy data with no correlation
X = np.zeros((20,2))
X[:,0] = np.random.randn(20)
X[:,1] = np.random.randn(20)

# Plot dummy data
plt.figure()
plt.plot(X[:,0], X[:,1], '.')
plt.title('Uncorrelated variables')
plt.xlabel(r'$X_1$')
plt.ylabel(r'$X_2$')
plt.show()
# Fit the Stan model
data = {'N' : X.shape[0], #number of datapoints
        'X' : X} #the data
fit = model_pc.sampling(data=data, iter=Niters, chains=Nchains, n_jobs=Nchains)
samples = fit.extract()

# Plot the posterior distribution for the correlation coefficient
plt.figure()
sb.distplot(samples['rho'])
plt.xlabel('Pearson Correlation Coefficient')
plt.ylabel('Posterior Probability')
plt.show()
# Create and plot dummy data w/ pooled correlation but no population correlation
N = 10
Ns = 5
X = np.zeros((N*Ns,2))
I = np.zeros(N*Ns, dtype=int) #invididual/group number
plt.figure()
for iS in range(Ns):
    x = np.random.rand(N)
    X1 = x + iS
    X2 = x * (np.floor(Ns/2)-iS) + iS + 0.2*np.random.randn(N)
    X[iS*N:iS*N+N,0] = X1
    X[iS*N:iS*N+N,1] = X2
    I[iS*N:iS*N+N] = iS+1
    plt.plot(X1, X2, '.')
plt.title('Pooled correlation but no population correlation!')
plt.xlabel(r'$X_1$')
plt.ylabel(r'$X_2$')
plt.show()
# Compute Frequentist estimates of the correlation
rho, pval = pearsonr(X[:,0], X[:,1])
print('Pooled correlation coefficient: %0.3g ( p = %0.3g )' % (rho, pval))
for iS in range(Ns):
    rho, pval = pearsonr(X[iS*N:iS*N+N,0], X[iS*N:iS*N+N,1])
    print('Individual %d\'s correlation coefficient: %0.3g ( p = %0.3g )' % ((iS+1), rho, pval))
# Data for Stan
data = {'N' : X.shape[0], #number of datapoints
        'X' : X}          #the data

# Fit the model
fit = model_pc.sampling(data=data, iter=Niters, chains=Nchains, n_jobs=Nchains)

# Get the MCMC samples (draws from the posterior distribution)
samples = fit.extract()

# Plot the posterior distribution for the correlation coefficient
plt.figure()
sb.distplot(samples['rho'])
plt.xlabel('Pearson Correlation Coefficient')
plt.ylabel('Posterior Probability')
plt.show()
#STAN code string for a two-level pearson correlation
MultilevelCorrelation = """
data {
    int<lower=0> N;    //number of datapoints
    int<lower=1> Ni;   //number of individuals/groups
    int<lower=1> I[N]; //individual of each datapoint
    vector[2] X[N];    //datapoints
}

parameters {
    vector[2] mu[Ni];               //per-individual mean
    vector<lower=0>[2] sig[Ni];     //per-individual std dev of each variable
    real<lower=-1,upper=1> rho[Ni]; //per-individual Pearson's rho
    real<lower=-1,upper=1> mu_rho;  //mean of population rhos
    real<lower=0> sig_rho;          //std dev of population rhos
}

transformed parameters {
    cov_matrix[2] C[Ni]; //covariance matrix for each individual
    for (i in 1:Ni) {
        C[i][1,1] = sig[i][1] * sig[i][1];
        C[i][1,2] = rho[i] * sig[i][1] * sig[i][2];
        C[i][2,1] = rho[i] * sig[i][1] * sig[i][2];
        C[i][2,2] = sig[i][2] * sig[i][2];
    }
}

model {
    // Each individual rho is drawn from population distribution
    rho ~ normal(mu_rho, sig_rho);
    
    // Each individual datapoint is drawn from its individual's distribution
    for (i in 1:N) {
        X[i] ~ multi_normal(mu[I[i]], C[I[i]]);
    }
}
"""
# Compile the Stan model for the multilevel correlation
model_ml = pystan.StanModel(model_code=MultilevelCorrelation)
# Data for Stan
data = {'N' : X.shape[0],        #number of datapoints
        'Ni': len(np.unique(I)), #number of individuals
        'I' : I,                 #subject of each datapoint
        'X' : X}                 #the datapoints

# Fit the model
fit = model_ml.sampling(data=data, iter=Niters, chains=Nchains, n_jobs=Nchains)

# Get the MCMC samples (draws from the posterior distribution)
samples = fit.extract()
# Plot the posterior distribution for the correlation coefficient
plt.figure()
sb.distplot(samples['mu_rho'])
plt.xlabel('Population Mean Pearson Correlation Coefficient')
plt.ylabel('Posterior Probability')
plt.show()
# Plot the per-subject posterior distributions for rho
plt.figure()
for iS in range(Ns):
    sb.kdeplot(samples['rho'][:,iS], shade=True)
plt.title('Per-Individual Posterior Probability Distributions of Rho')
plt.xlabel(r'$X_1$')
plt.ylabel(r'$X_2$')
plt.show()
# Plot the posterior joint distribution for the means
plt.figure()
cmaps = ["Blues", "Greens", "Reds", "Purples", "Oranges"]
for iS in range(Ns):
    sb.kdeplot(samples['mu'][:,iS,0], samples['mu'][:,iS,1], 
               shade=True, shade_lowest=False, cmap=cmaps[iS])
    plt.plot(X[iS*N:iS*N+N,0], X[iS*N:iS*N+N,1], '.')
plt.title('Per-Individual Joint Posterior Probability Distributions of the Means')
plt.xlabel(r'$X_1$')
plt.ylabel(r'$X_2$')
plt.show()
# Create and plot dummy data w/ no pooled correlation but a population correlation
N = 3
Ns = 5
X = np.zeros((N*Ns,2))
I = np.zeros(N*Ns, dtype=int) #invididual/group number
plt.figure()
for iS in range(Ns):
    x = np.random.rand(N)
    X1 = x + iS
    X2 = x + 0.2*np.random.randn(N) - 0.04*iS
    X[iS*N:iS*N+N,0] = X1
    X[iS*N:iS*N+N,1] = X2
    I[iS*N:iS*N+N] = iS+1
    plt.plot(X1, X2, '.')
plt.title('No Pooled correlation but a population correlation!')
plt.xlabel(r'$X_1$')
plt.ylabel(r'$X_2$')
plt.show()
# Compute Frequentist estimates of the correlation
rho, pval = pearsonr(X[:,0], X[:,1])
print('Pooled correlation coefficient: %0.3g ( p = %0.3g )' % (rho, pval))
for iS in range(Ns):
    rho, pval = pearsonr(X[iS*N:iS*N+N,0], X[iS*N:iS*N+N,1])
    print('Individual %d\'s correlation coefficient: %0.3g ( p = %0.3g )' % (iS+1, rho, pval))
# Data for Stan
data = {'N' : X.shape[0], #number of datapoints
        'X' : X} #the data

# Fit the non-multilevel model
fit = model_pc.sampling(data=data, iter=Niters, chains=Nchains, n_jobs=Nchains)

# Get the MCMC samples (draws from the posterior distribution)
samples = fit.extract()

# Plot the posterior distribution for the correlation coefficient
plt.figure()
sb.distplot(samples['rho'])
plt.xlabel('Pearson Correlation Coefficient')
plt.ylabel('Posterior Probability')
plt.show()
# Data for Stan
data = {'N' : X.shape[0], #number of datapoints
        'Ni': len(np.unique(I)), #number of individuals
        'I' : I, #subject of each datapoint
        'X' : X} #the datapoints

# Fit the multilevel model
fit = model_ml.sampling(data=data, iter=Niters, chains=Nchains, n_jobs=Nchains)

# Get the MCMC samples (draws from the posterior distribution)
samples = fit.extract()
# Plot the posterior distribution for the correlation coefficient
plt.figure()
sb.distplot(samples['mu_rho'])
plt.xlabel('Population Mean Pearson Correlation Coefficient')
plt.ylabel('Posterior Probability')
plt.show()
# Plot the per-subject posterior distributions for rho
plt.figure()
for iS in range(Ns):
    sb.kdeplot(samples['rho'][:,iS])
plt.title('Per-Individual Posterior Probability Distributions of Rho')
plt.xlabel('Pearson Correlation Coefficient')
plt.ylabel('Posterior Probability')
plt.title('Multilevel Model Per-Individual Estimates of Rho')
plt.show()
# Fit the non-multilevel model for each individual, individually
plt.figure()
for iS in range(Ns):
    tX = X[iS*N:iS*N+N,:] #data for this subject
    data = {'N' : tX.shape[0],
            'X' : tX}
    fit = model_pc.sampling(data=data, iter=Niters, chains=Nchains, n_jobs=Nchains)
    samples = fit.extract()
    sb.kdeplot(samples['rho'])
plt.xlabel('Pearson Correlation Coefficient')
plt.ylabel('Posterior Probability')
plt.title('Non-Multilevel Model Per-Individual Estimates of Rho')
plt.show()
# Plot normal distribution vs t-distribution
from scipy.stats import norm, t
x = np.arange(-10, 10, 0.001)
p_normal = norm.pdf(x,0,1)
p_tdist = t.pdf(x,1,0,1)

plt.figure()
nd = plt.plot(x, p_normal, label='normal dist')
td = plt.plot(x, p_tdist, label='t dist')
plt.title('Normal distribution vs t-distribution')
plt.legend()
plt.show()

plt.figure()
nd = plt.plot(x, p_normal, label='normal dist')
td = plt.plot(x, p_tdist, label='t dist')
plt.ylim((-0.005, 0.05))
plt.title('Zoomed in')
plt.legend()
plt.show()
#STAN code string for a robust two-level Pearson correlation
RobustCorrelation = """
data {
    int<lower=0> N;    //number of datapoints
    int<lower=1> Ni;   //number of individuals/groups
    int<lower=1> I[N]; //individual of each datapoint
    vector[2] X[N];    //datapoints
}

parameters {
    vector[2] mu[Ni];               //per-individual mean
    vector<lower=0>[2] sig[Ni];     //per-individual std dev of each variable
    real<lower=-1,upper=1> rho[Ni]; //per-individual Pearson's rho
    real<lower=-1,upper=1> mu_rho;  //mean of population rhos
    real<lower=0> sig_rho;          //std dev of population rhos
}

transformed parameters {
    cov_matrix[2] C[Ni]; //covariance matrix for each individual
    for (i in 1:Ni) {
        C[i][1,1] = sig[i][1] * sig[i][1];
        C[i][1,2] = rho[i] * sig[i][1] * sig[i][2];
        C[i][2,1] = rho[i] * sig[i][1] * sig[i][2];
        C[i][2,2] = sig[i][2] * sig[i][2];
    }
}

model {
    // Each individual rho is drawn from population distribution
    rho ~ normal(mu_rho, sig_rho);

    // Each individual datapoint is drawn from its individual's distribution
    for (i in 1:N) {
        X[i] ~ multi_student_t(1, mu[I[i]], C[I[i]]);
    }
}
"""
# Compile the Stan model for the robust multilevel correlation
model_rml = pystan.StanModel(model_code=RobustCorrelation)
# Create and plot dummy data w/ no pooled correlation but a population correlation
N = 5
Ns = 5
X = np.zeros((N*Ns,2))
I = np.zeros(N*Ns, dtype=int) #invididual/group number
plt.figure()
for iS in range(Ns):
    x = np.random.rand(N)
    X1 = x + iS
    X2 = x + 0.1*np.random.randn(N) - 0.04*iS
    if iS==0: #individual 1 has an outlier measurement
        X1[0] = 3
        X2[0] = -3
    elif iS==1: #so does individual 2
        X1[0] = 4
        X2[0] = -4
    X[iS*N:iS*N+N,0] = X1
    X[iS*N:iS*N+N,1] = X2
    I[iS*N:iS*N+N] = iS+1
    plt.plot(X1, X2, '.')
plt.title('Positive population correlation, but with outliers!')
plt.xlabel(r'$X_1$')
plt.ylabel(r'$X_2$')
plt.show()
# Fit the Stan model
data = {'N' : X.shape[0], #number of datapoints
        'Ni': len(np.unique(I)), #number of individuals
        'I' : I, #subject of each datapoint
        'X' : X} #the datapoints
fit = model_rml.sampling(data=data, iter=Niters, chains=Nchains, n_jobs=Nchains)
samples = fit.extract()

# Plot the posterior distribution for the population correlation coefficient
plt.figure()
sb.distplot(samples['mu_rho'])
plt.title('Robust Estimate of Rho across the Population')
plt.xlabel('Population Mean Pearson Correlation Coefficient')
plt.ylabel('Posterior Probability')
plt.show()

# Plot the per-subject posterior distributions for rho
plt.figure()
for iS in range(Ns):
    sb.kdeplot(samples['rho'][:,iS])
plt.title('Per-Individual Posterior Probability Distributions of Rho')
plt.xlabel('Pearson Correlation Coefficient')
plt.ylabel('Posterior Probability')
plt.title('Robust Per-Individual Estimates of Rho')
plt.show()
# Fit the Stan model
data = {'N' : X.shape[0], #number of datapoints
        'Ni': len(np.unique(I)), #number of individuals
        'I' : I, #subject of each datapoint
        'X' : X} #the datapoints
fit = model_ml.sampling(data=data, iter=Niters, chains=Nchains, n_jobs=Nchains)
samples = fit.extract()

# Plot the posterior distribution for the population correlation coefficient
plt.figure()
sb.distplot(samples['mu_rho'])
plt.title('NON-Robust Estimate of Rho across the Population')
plt.xlabel('Population Mean Pearson Correlation Coefficient')
plt.ylabel('Posterior Probability')
plt.show()

# Plot the per-subject posterior distributions for rho
plt.figure()
for iS in range(Ns):
    sb.kdeplot(samples['rho'][:,iS])
plt.title('Per-Individual Posterior Probability Distributions of Rho')
plt.xlabel('Pearson Correlation Coefficient')
plt.ylabel('Posterior Probability')
plt.title('NON-Robust Per-Individual Estimates of Rho')
plt.show()
# Load Massachusetts Public School data
df = pd.read_csv('../input/MA_Public_Schools_2017.csv')
# Take a look at the data
df.sample(n=5)
# Plot distribution of Expenditures per Pupil
costs = df['Average Expenditures per Pupil']
sb.distplot(costs.dropna())
plt.show()

# Print the number of empty entries for this column
print('Number of nan Expenditures', costs.isnull().sum())
print('Percent nan Expenditures', 100*costs.isnull().sum()/df.shape[0])
# Plot distribution of progres and performance
ppi = df['Progress and Performance Index (PPI) - All Students']
sb.distplot(ppi.dropna())
plt.show()

# Print the number of empty entries for this column
print('Number of nan PPI', ppi.isnull().sum())
print('Percent nan PPI', 100*ppi.isnull().sum()/df.shape[0])
# Group schools by 1st 3 digits of zip code
df['ZipGroup'], _ = pd.factorize(df['Zip'].apply(lambda x: ("%05d" % x)[:3]))
df['ZipGroup'] = df['ZipGroup'] + 1 #start indexing at 1

# Plot the number of schools per zip group
plt.figure()
sb.countplot(x='ZipGroup', data=df)
plt.title('Number of Schools per Zip Group')
plt.show()
# Plot school expenditures per pupil against the performance index
plt.figure()
plt.plot(costs, ppi, '.')
plt.xlabel('Average Expenditures per Pupil')
plt.ylabel('Progress and Performance Index')
plt.title('Costs per student vs Studen Performance')
plt.show()
# Compute poooled correlation w/ frequentist pval
k = ~np.isnan(costs) & ~np.isnan(ppi)
rho, pval = pearsonr(costs[k], ppi[k])
print('Pooled correlation coefficient: %0.3g ( p = %0.3g )' % (rho, pval))
#STAN code string for a robust two-level Pearson correlation with priors!
RobustCorrelation = """
data {
    int<lower=0> N;    //number of datapoints
    int<lower=1> Ni;   //number of individuals/groups
    int<lower=1> I[N]; //individual of each datapoint
    vector[2] X[N];    //datapoints
}

parameters {
    vector[2] mu;                   //mean
    vector<lower=0>[2] sig;         //std dev of each variable
    real<lower=-1,upper=1> rho[Ni]; //per-individual Pearson's rho
    real<lower=-1,upper=1> mu_rho;  //mean of population rhos
    real<lower=0> sig_rho;          //std dev of population rhos
}

transformed parameters {
    cov_matrix[2] C[Ni]; //covariance matrix for each individual
    for (i in 1:Ni) {
        C[i][1,1] = sig[1] * sig[1];
        C[i][1,2] = rho[i] * sig[1] * sig[2];
        C[i][2,1] = rho[i] * sig[1] * sig[2];
        C[i][2,2] = sig[2] * sig[2];
    }
}

model {    
    // Each individual rho is drawn from population distribution
    rho ~ normal(mu_rho, sig_rho);

    // Each individual datapoint is drawn from its individual's distribution
    for (i in 1:N) {
        X[i] ~ multi_student_t(1, mu, C[I[i]]);
    }
}
"""
# Compile the Stan model for the robust multilevel correlation
model_rml = pystan.StanModel(model_code=RobustCorrelation)
# Only keep schools which have both expenditures and performance data
datacols = ['Average Expenditures per Pupil',
            'Progress and Performance Index (PPI) - All Students']
sdf = df.dropna(subset=datacols)

# Normalize data
sdf[datacols] = sdf[datacols].apply(zscore)
# Fit the Stan model
data = {'N' : sdf.shape[0], #number of schools
        'Ni': len(sdf['ZipGroup'].unique()), #number of zip groups
        'I' : sdf['ZipGroup'], #zip group index of each school
        'X' : sdf[datacols].values} #the datapoints
fit = model_rml.sampling(data=data, iter=Niters, chains=Nchains, n_jobs=Nchains)
samples = fit.extract()
# Plot the posterior distribution for the across-group mean correlation coefficient
plt.figure()
sb.distplot(samples['mu_rho'])
plt.title(r'Robust Estimate of $\mu_\rho$ across the entire state of Massachusetts')
plt.xlabel('Group Mean Pearson Correlation Coefficient')
plt.ylabel('Posterior Probability')
plt.show()
print("%0.2g%% of posterior for rho is >0" % 
      (100*np.sum(samples['mu_rho']>0)/samples['mu_rho'].shape[0]))
print("%0.2g%% of posterior for rho is >0.1" % 
      (100*np.sum(samples['mu_rho']>0.1)/samples['mu_rho'].shape[0]))
# Plot the per-group posterior distributions for rho
plt.figure()
for iS in range(len(sdf['ZipGroup'].unique())):
    sb.kdeplot(samples['rho'][:,iS])
plt.title('Posterior Probability Distributions of Rho for each ZIP group')
plt.xlabel('Pearson Correlation Coefficient')
plt.ylabel('Posterior Probability')
plt.title('Robust Per-Group Estimates of Rho')
plt.show()
# Plot t-distributions with different degrees of freedom
from scipy.stats import norm, t
x = np.arange(-5, 5, 0.001)
plt.figure()
for iNu in [1, 2, 5, 100]:
    p_tdist1 = t.pdf(x,iNu,0,1)
    plt.plot(x, p_tdist1, label=r'$\nu='+str(iNu)+r'$')
p_normal = norm.pdf(x,0,1)
nd = plt.plot(x, p_normal, '--', label='Normal dist')
plt.xlim((-5, 5))
plt.xlabel('Value')
plt.ylabel('Probability')
plt.title(r't-distributions with different $\nu$ values')
plt.legend()
plt.show()