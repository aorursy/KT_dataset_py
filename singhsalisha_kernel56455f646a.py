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
import pandas as pd

boxes_split1 = pd.read_csv("../input/google-landmarks-dataset/boxes_split1.csv")

boxes_split2 = pd.read_csv("../input/google-landmarks-dataset/boxes_split2.csv")

index = pd.read_csv("../input/google-landmarks-dataset/index.csv")

recognition_solution = pd.read_csv("../input/google-landmarks-dataset/recognition_solution.csv")

retrieval_solution = pd.read_csv("../input/google-landmarks-dataset/retrieval_solution.csv")

test = pd.read_csv("../input/google-landmarks-dataset/test.csv")

train = pd.read_csv("../input/google-landmarks-dataset/train.csv")
import pandas as pd

boxes_split1 = pd.read_csv("../input/google-landmarks-dataset/boxes_split1.csv")

boxes_split2 = pd.read_csv("../input/google-landmarks-dataset/boxes_split2.csv")

index = pd.read_csv("../input/google-landmarks-dataset/index.csv")

recognition_solution = pd.read_csv("../input/google-landmarks-dataset/recognition_solution.csv")

retrieval_solution = pd.read_csv("../input/google-landmarks-dataset/retrieval_solution.csv")

test = pd.read_csv("../input/google-landmarks-dataset/test.csv")

train = pd.read_csv("../input/google-landmarks-dataset/train.csv")
import pandas as pd

boxes_split1 = pd.read_csv("../input/google-landmarks-dataset/boxes_split1.csv")

boxes_split2 = pd.read_csv("../input/google-landmarks-dataset/boxes_split2.csv")

index = pd.read_csv("../input/google-landmarks-dataset/index.csv")

recognition_solution = pd.read_csv("../input/google-landmarks-dataset/recognition_solution.csv")

retrieval_solution = pd.read_csv("../input/google-landmarks-dataset/retrieval_solution.csv")

test = pd.read_csv("../input/google-landmarks-dataset/test.csv")

train = pd.read_csv("../input/google-landmarks-dataset/train.csv")
import pandas as pd

boxes_split1 = pd.read_csv("../input/google-landmarks-dataset/boxes_split1.csv")

boxes_split2 = pd.read_csv("../input/google-landmarks-dataset/boxes_split2.csv")

index = pd.read_csv("../input/google-landmarks-dataset/index.csv")

recognition_solution = pd.read_csv("../input/google-landmarks-dataset/recognition_solution.csv")

retrieval_solution = pd.read_csv("../input/google-landmarks-dataset/retrieval_solution.csv")

test = pd.read_csv("../input/google-landmarks-dataset/test.csv")

train = pd.read_csv("../input/google-landmarks-dataset/train.csv")
import pandas as pd

boxes_split1 = pd.read_csv("../input/google-landmarks-dataset/boxes_split1.csv")

boxes_split2 = pd.read_csv("../input/google-landmarks-dataset/boxes_split2.csv")

index = pd.read_csv("../input/google-landmarks-dataset/index.csv")

recognition_solution = pd.read_csv("../input/google-landmarks-dataset/recognition_solution.csv")

retrieval_solution = pd.read_csv("../input/google-landmarks-dataset/retrieval_solution.csv")

test = pd.read_csv("../input/google-landmarks-dataset/test.csv")

train = pd.read_csv("../input/google-landmarks-dataset/train.csv")
import pandas as pd

boxes_split1 = pd.read_csv("../input/google-landmarks-dataset/boxes_split1.csv")

boxes_split2 = pd.read_csv("../input/google-landmarks-dataset/boxes_split2.csv")

index = pd.read_csv("../input/google-landmarks-dataset/index.csv")

recognition_solution = pd.read_csv("../input/google-landmarks-dataset/recognition_solution.csv")

retrieval_solution = pd.read_csv("../input/google-landmarks-dataset/retrieval_solution.csv")

test = pd.read_csv("../input/google-landmarks-dataset/test.csv")

train = pd.read_csv("../input/google-landmarks-dataset/train.csv")
import pandas as pd

boxes_split1 = pd.read_csv("../input/google-landmarks-dataset/boxes_split1.csv")

boxes_split2 = pd.read_csv("../input/google-landmarks-dataset/boxes_split2.csv")

index = pd.read_csv("../input/google-landmarks-dataset/index.csv")

recognition_solution = pd.read_csv("../input/google-landmarks-dataset/recognition_solution.csv")

retrieval_solution = pd.read_csv("../input/google-landmarks-dataset/retrieval_solution.csv")

test = pd.read_csv("../input/google-landmarks-dataset/test.csv")

train = pd.read_csv("../input/google-landmarks-dataset/train.csv")
import pandas as pd

boxes_split1 = pd.read_csv("../input/google-landmarks-dataset/boxes_split1.csv")

boxes_split2 = pd.read_csv("../input/google-landmarks-dataset/boxes_split2.csv")

index = pd.read_csv("../input/google-landmarks-dataset/index.csv")

recognition_solution = pd.read_csv("../input/google-landmarks-dataset/recognition_solution.csv")

retrieval_solution = pd.read_csv("../input/google-landmarks-dataset/retrieval_solution.csv")

test = pd.read_csv("../input/google-landmarks-dataset/test.csv")

train = pd.read_csv("../input/google-landmarks-dataset/train.csv")