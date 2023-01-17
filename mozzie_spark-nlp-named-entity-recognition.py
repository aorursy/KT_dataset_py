# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import os

# Install java
! apt-get install -y openjdk-8-jdk-headless -qq > /dev/null
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
os.environ["PATH"] = os.environ["JAVA_HOME"] + "/bin:" + os.environ["PATH"]
! java -version

# Install pyspark
! pip install --ignore-installed pyspark==2.4.5 spark-nlp==2.5.1

import sparknlp

spark = sparknlp.start()

print("Apache Spark version")
spark.version
print("Spark NLP version")
sparknlp.version()

from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline('recognize_entities_dl', 'en')

result = pipeline.annotate('Donald John Trump (born June 14, 1946) is the 45th and current president of the United States.')

result
print(result['ner'])

ner = [result['ner'] for content in result]
token = [result['token'] for content in result]
# let's put token and tag together
list(zip(token[0], ner[0]))

print(result['entities'])
