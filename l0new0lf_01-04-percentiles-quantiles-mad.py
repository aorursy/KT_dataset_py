import numpy as np
mu, sigma = 0, 0.1 # mean and standard deviation

data = np.random.normal(mu, sigma, 50)



print("Input shape: ", data.shape)

print("\nInput Data: ", data)
print('0th percentile i.e least most val in data is: ', np.percentile(data,0))

print('90th percentile i.e 90% of data lie below: ', np.percentile(data,90), "(and above 0th percentile)")
print(np.percentile(data, np.arange(0, 100, 25)))


from statsmodels import robust

print("MAD: ", robust.mad(data))