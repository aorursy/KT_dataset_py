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
def submission_generation(dataframe, name):

    """

    Esta función genera un csv a partir de un dataframe de pandas. 

    Con FileLink se genera un enlace desde el que poder descargar el fichero csv

    

    dataframe: DataFrame de pandas

    name: nombre del fichero csv

    """

    import os

    from IPython.display import FileLink

    os.chdir(r'/kaggle/working')

    dataframe.to_csv(name, index = False)

    return  FileLink(name)
train_raw = pd.read_csv("/kaggle/input/cancer-de-mama/train.csv")

test_raw = pd.read_csv("/kaggle/input/cancer-de-mama/test.csv")



train_raw.head()
answer_everybody_survive = pd.DataFrame({"SampleID": test_raw.SampleID, 

                                      "Class": 2})
submission_generation(answer_everybody_survive, "submission_zero.csv")