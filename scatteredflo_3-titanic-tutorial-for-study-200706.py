import pandas as pd

import numpy as np

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# kaggle dataset 확인
# 직접 입력해 보세요!
# 직접 입력해 보세요!
# 직접 입력해 보세요!
# 직접 입력해 보세요!
# 직접 입력해 보세요!
# 직접 입력해 보세요!
# 직접 입력해 보세요!
# 직접 입력해 보세요!
# 직접 입력해 보세요!
# 직접 입력해 보세요!
# 직접 입력해 보세요!
# 직접 입력해 보세요!
# 직접 입력해 보세요!
# 직접 입력해 보세요!
# 직접 입력해 보세요!
# 직접 입력해 보세요!
# 직접 입력해 보세요!
# 직접 입력해 보세요!
# 직접 입력해 보세요!
# 직접 입력해 보세요!
# 직접 입력해 보세요!
# 직접 입력해 보세요!
# 직접 입력해 보세요!
# 직접 입력해 보세요!
# 직접 입력해 보세요!
# 직접 입력해 보세요!
# 직접 입력해 보세요!
# 직접 입력해 보세요!
# 직접 입력해 보세요!
# 직접 입력해 보세요!
# 직접 입력해 보세요!
# 직접 입력해 보세요!
# 직접 입력해보세요!

# 직접 입력해보세요!

# 직접 입력해보세요!

# 직접 입력해 보세요!
# 직접 입력해 보세요!
# 직접 입력해 보세요!
# 직접 입력해 보세요!
# 직접 입력해 보세요!
# 직접 입력해 보세요!
# 직접 입력해 보세요!
# 직접 입력해 보세요!
# 직접 입력해 보세요!
# 직접 입력해 보세요!
# 직접 입력해 보세요!
# 직접 입력해 보세요!
# 직접 입력해 보세요!
# 직접 입력해 보세요!
# 직접 입력해 보세요!
# 직접 입력해 보세요!
# 직접 입력해 보세요!
# 직접 입력해 보세요!
# 직접 입력해 보세요!
# 직접 입력해 보세요!
# 직접 입력해 보세요!
# 직접 입력해 보세요!
# 직접 입력해 보세요!
!pip install pydotplus

from sklearn.externals.six import StringIO  

from IPython.display import Image  

from sklearn.tree import export_graphviz

import pydotplus

dot_data = StringIO()

export_graphviz(model, out_file=dot_data,  

                filled=True, rounded=True,

                special_characters=True)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  

Image(graph.create_png())