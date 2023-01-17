# This R environment comes with all of CRAN preinstalled, as well as many other helpful packages
# The environment is defined by the kaggle/rstats docker image: https://github.com/kaggle/docker-rstats
# For example, here's several helpful packages to load in 

#library(ggplot2) # Data visualization
#library(readr) # CSV file I/O, e.g. the read_csv function

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#system("ls ../input")

import pandas as pd

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="white", color_codes=True)

list = [7991027,13684736,61690]

series = pd.Series(list, index=['P', 'R', 'O'], name='Posts Stackoverflow')
#series.plot(kind='pie', figsize=(6, 6))
series.plot(kind='pie', labels=['Preguntas', 'Respuestas', 'Otros'], colors=['#207ce5','#dbdbdb', 'g'],autopct='%.2f', fontsize=20, figsize=(10, 10))

# Any results you write to the current directory are saved as output.
list = [5549458,2441569]

series = pd.Series(list, index=['CT', 'T'], name='Posts Stackoverflow')
#series.plot(kind='pie', figsize=(6, 6))
series.plot(kind='pie', labels=['Lenguaje natural + Lenguaje codigo', 'Lenguaje natural'], colors=['#207ce5','#dbdbdb'],autopct='%.2f', fontsize=16, figsize=(10, 10))

import numpy as np
import matplotlib.pyplot as plt

lista = np.array(range(0,100,5))

random = [1.535737812911725647e-02,
4.756669960474307207e-02,
7.542545015371102113e-02,
9.708498023715413094e-02,
1.134387351778656100e-01,
1.271890096618357446e-01,
1.392174854131375816e-01,
1.495604825428195062e-01,
1.590533962816571556e-01,
1.670446310935441159e-01,
1.738359384357407722e-01,
1.802035298638559324e-01,
1.860412232694841350e-01,
1.913978684359119165e-01,
1.969120553359683667e-01,
2.019402585638998548e-01,
2.065372394016895541e-01,
2.108379995608256341e-01,
2.148620067956452384e-01,
2.189241600790513553e-01]

texto = [7.402915019762845716e-01,
7.708045125164689759e-01,
7.923281730346947160e-01,
8.056241765480895989e-01,
8.142786561264822032e-01,
8.202898550724637694e-01,
8.247694334650855774e-01,
8.281692605401844709e-01,
8.308602327624067252e-01,
8.330377140974967176e-01,
8.348417475146724387e-01,
8.363554018445323868e-01,
8.376488547684199926e-01,
8.387575287031808768e-01,
8.397183794466402951e-01,
8.405591238471672444e-01,
8.413009571417499055e-01,
8.419603645147123450e-01,
8.425555613341654260e-01,
8.430912384716731101e-01]

codigo = [2.206851119894598090e-01,
2.749423583662713999e-01,
3.142649319279753883e-01,
3.452219202898550443e-01,
3.724571805006587355e-01,
3.960364514712340633e-01,
4.172571993224167275e-01,
4.364799489459815218e-01,
4.543780193236715559e-01,
4.702173913043478648e-01,
4.846291472032578374e-01,
4.977375658761528099e-01,
5.097623391101652190e-01,
5.209774374176547873e-01,
5.314212779973649381e-01,
5.411736248353097301e-01,
5.501472525769200983e-01,
5.586293185477967382e-01,
5.666064246584840980e-01,
5.740386198945981455e-01]

multimodal = [5.467638339920949386e-01,
6.536561264822133843e-01,
7.113636363636363091e-01,
7.437767621870883250e-01,
7.642786561264821588e-01,
7.782540074659639595e-01,
7.884704968944098447e-01,
7.963109354413702157e-01,
8.024493119601814328e-01,
8.074258893280632510e-01,
8.115470415618636357e-01,
8.149813350900305675e-01,
8.178999442586398771e-01,
8.204104319593449324e-01,
8.225861879666226395e-01,
8.244976943346506992e-01,
8.261901302022783833e-01,
8.276945176401697690e-01,
8.290405485056513424e-01,
8.302519762845849138e-01]

plt.figure()
plt.plot(lista, texto, 'b', linewidth = 2, label = 'Texto')
plt.plot(lista, multimodal, 'g', linewidth = 2, label = 'Multimodal')
plt.plot(lista, codigo, 'r', linewidth = 2, label = 'Codigo')
plt.plot(lista, codigo, 'm', linewidth = 2, label = 'Aleatorio')

plt.xlabel("Etiquetas sugeridas", fontsize = 18)
plt.ylabel("Recall", fontsize = 18)
plt.legend(loc = 4, fontsize = 12)