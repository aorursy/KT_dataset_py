# This Python 3 environment comes with many helpful analytics libraries installed

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

! pip install --ignore-installed pyspark==2.4.4
import pandas as pd



from pyspark.sql import SparkSession

import pyspark.sql.functions as F

import pyspark.sql.types as T

from pyspark.ml.feature import *
hc = (SparkSession.builder

                  .appName('Spark Word2Vec Model')

                  .enableHiveSupport()

                  .config("spark.executor.memory", "4G")

                  .config("spark.driver.memory","18G")

                  .config("spark.executor.cores","7")

                  .config("spark.python.worker.memory","4G")

                  .config("spark.driver.maxResultSize","6G")

                  .config("spark.sql.crossJoin.enabled", "true")

                  .config("spark.serializer","org.apache.spark.serializer.KryoSerializer")

                  .config("spark.kryoserializer.buffer.max","1024M")

                  .getOrCreate())
hc.version
from pyspark.ml import Pipeline, PipelineModel



pipeLineWord2VecModel = PipelineModel.load("/kaggle/input/apache-spark-word2vec-model/multivac_word2vec_ml_200k/")

pipeLineWord2VecModel.stages
word2VecModel = pipeLineWord2VecModel.stages[0]

word2VecModel.findSynonyms("climate change", 10).show()
word2VecModel.findSynonyms("football", 10).show()

word2VecModel.findSynonyms("cancer", 10).show()

word2VecModel.findSynonyms("london", 10).show()
