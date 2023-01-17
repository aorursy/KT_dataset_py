import requests # This library is used to make requests to internet

# We are storing url of dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
# We are creating a requests variable with the above url
r = requests.get(url, allow_redirects=True)
# We are writing the content of above request to 'iris.data' file
open('iris.data', 'wb').write(r.content)

# Now, we have our files downloaded 
# Printing the content of file
iris = open('iris.data','r')
for lines in iris:
    print(lines)
import os
print(os.listdir('./')) # This will print the content of current directory
print(os.listdir('../input')) # This will print the content of input directory
import os

print(os.listdir('./')) # This will print the content of current directory
os.remove('iris.data') # This will remove file 'iris.data'
print(os.listdir('./')) # This will print the content of current directory