#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAT8AAACeCAMAAABzT/1mAAAB2lBMVEVPr6g1fIia2N70sE8fMjNaq0b///8qfI6Rd065dSRQsqu2dykzd4Xyr07fnENHsKrutU/QjznxqEe1ag9TubJemZaf2Nub0dM/ZGQbIiTU8/IpYmUcJigbHyEdLC0eLjBZoZw6hY5LqKQ9i5FBk5a45OQAAABFnJxEk44vW1lAiINJoJpHm5U3cm8nRkUjOzs1bGk7fHgAIi3I7e4vXVvv+vkAHR8nRURJWlspZGlFbXBPi5Hj8O+KrbEkcHy10NKlw8QAExbt7u7T4t5OkPWFxcHCklxcr6sXCxB1v5FWqDlwnKJ3qKzC2Nt7jXied0h3eF4AZHDAoV2XmXRtjYPcqVKznWGRuLZEgY92iXmglmojeJFchILLqFr+t0ndkzLTiCixi1aml2+dn4idrqCdwLXIkkujuKqVytI7SUfAxsWPlpWusrFUY2MAJydlenqaoqMXJhw3Z5lOpsIyVX4dKhZAc7QrSGcYGQAsEh+Hen5bNj+fkZpKitlRYlQtNS25Vm0xV25TPjaUUWOXemROSzRPkvs6Y4dgUT8LOjmCazxOSks6PjHqt2/q0q3kaYjl2sKBU2Dbbo3pw4pnTUF+xq6Jz8Rrt3NZpFFxuH6lqXRqpZNSsJZTsYsZWxqAAAAWW0lEQVR4nO2d+WPbRnbHqRiBkqlk2g4sCTQOGY4kngApkaJsUdTB0NRWUdIcjryR7DRt160oWbaba9s06222aZtzu9nmbPZ/7XszAIiLl02IlMT3iw+QAOaD7zvmzZCMREhEm7s2MjY2MrQnsAgaIYQxHELs1iKWAcPIUIddW8RpTIdXhzrs3CI+Ax1qLw4Zdmh+fg2GQ2dub8H8hgw7teb8LIbDpNLKWvOzk8pQh02sPT+b4VCHAdYZvyHDZtY5P5NhZO7F4USlYd3xMyEOJyq2PQE/U4fDIhvtyfhZOhwyfAp+DYZn2Jmfkt+ZZ9gDfowhFtln0Jd7xM9meNYScw/5uRieFYg95mczfPGM6DAEfibDSOos6DAkfjbD014ghsjPhHi6i+yw+VkMT2sH8Tj4MYanswt7XPxshqesQDxOfqeR4XHzMxlG6DJ9vwffA+sHvwbDk18g9oufxTB1wlcD+snPhBiZO8EFYr/xobHVgJOpw36zs+2EdmH7jc1lx8lwrFzuxWn6jcxnIReI9LzlkeWtndWdgEuM7u6OdnW+fuMKtDD2wo6h5JDb1tjLhUKB314t+AW4+8qrr/5NVwD7jaqp9UiHDNvIyNbKVnmlUOJL21vllyvl5dJr5YL3taO3X7/7/PPPnQ5+aE/FEAPc1a2V5fJaqXR7q1BYHhlZLm8sl8dGtkrlq/xYubTsprf7xvi5c391mvihPelWh7GdSqlQKG2sgda2y+XSFqqwXHkZTrO8XR5DnK4z7r556+65U8iPmjVh7oLh2NZaaaT82so2cCutrpRosCtvr1GMI8DT9erRsbeQ3mnlh9atDseWIUGsXkXBjZR45qvltW2WNMqVFcd5xnZfGT937pTzQ+uuQORXK6UyrVdWeaq7kbGVCvvztdW1xut29+6g+M6ffn5oHe/JLhd2lkv0dVuFkQJzXMi7V+HPrUqBL+wwKY6OUtcdP39x/EzwQ+soMWPOWMYsC85avs2vQUhcrfClAnj08sry1TJ78+4rF1B358+fO392+KG1ZQjYXiu/BhB3+NWxlQK/Ut5a21q+TbmZbxrFkg/Fh9o7Y/zQWjJcrpRWSyXAtrq2gjjHHB2e0dHd0dGxvV+/PU5dl4a+M8gPrdme7PLq9trK1jJIrVy2D42OIrnc3htvvX7rwvhdWrOMn2fJ94zyoxbUhS27ue3ujt7ee+WNt+7cunAX7NI771x0iO+M80ML7GRTR7299+Yrb79+B7ldAm5/e/Hmu3/39//wm0kMeuetyu/M80NzFIgguJHlPcpt3NTbvXuIbWfyGWYTl8416A35WYZJBRiajkr1xuT2jMeG/JoaIdELly7eA27/+Jt/8nIb8uvAohcuvtAM3JBfe0N+Po8d8uvYhvyezob8ns5OBz8iCAIJ+RrBFsRvYmbiZPEjScPIpoRwLxJsAfxmYrHYxInip8kcJ8m6Fu5VAs3HbyJm2sSJ4Uc4lVP3F9eNPriwj99CzAtw0PmRFMhvtrKxMRnmVZqYl1/MYf3hR7rNBCQlcov7fL2+0AcHbsFvoS/8SDIXJ92lAk08Olji+cPCAPCbmLGtL/FPyMiKKGVIVxqU9vn7/MYhXws+HHAuEtEICb4I6L8LDxiw+o+kRY7jFtcXU10AFLj1Sn2D5/lq4Ckj6YgPiCbJ8XgyHaBzklS5RLzj5zdg/ARd5ZT7hwcHnb6Bvuno/gZfOYhpQfqD6CiLetxNiiRlThIl2Uj5Xo61kCJzyQ4BDhq/IqivztdLnb6BjmEVtMfX96JBB0lRRUEX3YIm8JyA0+wDLyZiQC20ZBiJE8pPVzgFxNQslAWaVkB+/EwQPyErcdz+wf37aRcQIa8A1aXDuvc6KL/1Cr8x02EuGjR+idnKwwp/f8MXyjAnN4vrdSj+eD6w/BMSKnf0gN8ouEEJIDMO/rvivY4GtZABtVApUMx+GzR++dn6YT1ATCRdlCQjHgBQEJKz64uKsTQdeMKEujgLAcHzQITiev3gkK/UfddRaS10wHd4w5TfxMTA8MvMPsS7n/SKKbUJglE20x6ARNAyHKZs7tE/vxd4wuz6w8ND/oFnciJk1g/qgJX31tyCsQgvhgOBudxvyG+7USv3mx+5fAjiO3zow4DxXrl/P+HKo0SI66LEUXv0fjC/zGy9fshv8J6AJhjwnPiDGd/rE+uHGxWIwB1OBpFfBScbsZlB4BeZoKmg7os+SGnx4UbFSYHkDFnlTPvgw8CSg+RmobSu87EF75EYzxcmfVkC9FrhQYG+wqaJWfzAmgE8Tn7RGcrPG71pi4Bb3zisONIAFnGcKooGc9+PgvnFoTbkH9aJlxRcqRaUY1/GQpzvuP508IsNAD9B31968PBwx0sBpyWLD/nKhsPhIIeqip7TUnDs0Ucf/jYXmJ2r9IEc+i6UrPBBKdZ8gB1PpR38msXAY/VfSVWUxX/5V+/ECvWnAgZHkULiImfgZDUnccpHn3wgNfg5Bh/Mg5DE7INARtGdChZDpW74Qf5FG4D4R6YUdMaP/80rDSjLlPvgVtXGAZiqyJiOoRR+xH1S5JSMxa+6YCdPIXID5FzfqLiuAsgVbnEj4PpCbnF9fXFR/V2nHaBBqv9IZGmf5oKPD7xlCvjo+iG/4CgqSFxWaDYWio8+eun9R5ySFRiCCMjNfA3JisoREJn9neOEQtwQcU730Dv3IHiE3sDjOx2Wz4PEj6SWlmguUH9vpYLJKvMjIQM+6o73QlEV2d9+++8vvf8BVDd5AeEl9SWTH8rMLG4+fSvauIouixzMSrjHHn5C2pDNZD5/bw7QTBTad2QHhx9JLi0t4c0f/eFjc55RMltSmGkhezjHC/KTMig4kv7wpf/6AKknBBJPSJI0uzZTo2JihTWa8h9R+yqywuWEJBySXbMZQUtgMUTrIfWzV6/AuZt1xAaSH0l+Cfxo+GNxDUZUsAYQzyrKEuVXM0cEBbVKh6998p8UH8cV85KoKHIijqyEtI5ikiiP+VcfW6iQe0ogSRCmkhMi2qT1UPIiXFsRsVvDzd/6TCKR6IIVB04CP+HG0tJUzuSXQnowFS6g/8IcLQv/v34I6LQSm3GRtCwzJ3/vJRMfjF6VpKxGmwxCHsSkylwSHfXov59bbDyntCinhDjyg4A5ybOJLuEkKCXVLIFkDvK7NS8lCYn7ujNd85tYmDku/4XodwPC3GMAIWroTYqsEzZHk2kc24faT1tgFYzwuaSzFPme8cjEp4pG0lo5IeCfkh4XBDwg3ZwXG4GMaIqsYT0JATM6ycfY/yqcUoR3Cwl4UvN3vlA5A/4+m22bhlvzw6X0GR+/S5fC4CdMLS0Vj9RvYQBSKi8ripTDu9eKsiwm4jh9exNwputUkZH6vtlK0Kwgh7jsgCYkVBH/iUePbn7qinUkwilx5Jsj+pdmgoK6s4jvFjCBfXVBwRgiKGJQv6cLfqyuLnn4ff3Nn0PJH/GlpaNP//iHI2ShqHKervtAvC8mI0xHhgB+uQ5ZNqfvQGQk7F1mioVs4hgsSYoKpBeMc/M3n5vnJOfkjhBD2t+HqYsoxs28okkqa02g/O7dmwd1JlKi1H4VphU/ayndze+d/6lUxkOpX6b0j//4+yMWAYvmdhYS38S1C6oLCf4JQZ8Q6XOY4x9QcWBlw/h5pm8oKKyt52++ijQytidGUYFr/IZibOr20lxKYtkc54JfnftMxW5PXlHa33QLfvZODg+/b8Lil/6TiU9Vk5aYSHoTpUP7V5CWU5uZTDy+CbOyUoI2A+kBys/dPoBAJiL2+Zs38ZRYHDKrzmAGr/H8/c0GU6jPKX+cE87fuzWvJmg60tvPQlrpb2aBmYvf3Vth6S8S/fZbln+NhisCvzgx1ytgPDnp+hWw3FyNRDbpccUMf6KbH5DA6DV/8+vHCBjeSiA7EA1rIjzr/YPFZIMOSUk01qGaH9/9Yl7NYx5uQH8yfpYOnfwu/blUr4fEr/Yn5ryc7hwZ5Ueom8rpokH5gWWyOlJOWVMGb6wnCtQn0ze//orp00glM3mdw+425be+6GzFQlEjUjUnlPl3zz3GWlzyh4Se8KtUKoch8dMWzULEOTSTHyQCtagYnG7xu3Ll+vWcRpJiE36gWEP4+ev/PbKOS4qiiPLUJPqvkFUUZ22CasVEDJPCr85B9oA7gAcmelcLesQvJP1h8GF1sNNvNMYPjonEUJUrTkskhKztv57BwkRD1N791MLHKZLMJZIsoWI71p2R47JCs72ifnH3M5U+QZGTO7jprucf4fmvnUpd0ojMIhhsoCowbNXQE9ctCV4v5u30wYneWoOI6/Ej66hSzCY1iH9mzago5hzR5id9jpWMJqpf0HQNc+mkI2f3kF949UsjlWacKB7gUgQ2sAwi5AxOVZQH9bUrCPG6lBQkS1+St1VCpjf4+joLCLprf7RwI6EXVdeLa3xp0vQAlbN6OZ1sQeiUXwPgO99sh+S/liu64naNn2H8sJpgiI0HB5yeuHL9ihxJyzY//wkh1bKGhLsQIXNTYDdc4oJ6pqRF6OzXF0F6wG/8PDPEeP7WN9+EMv+wWbj4TfKFKM0fGJPMyZqyCDJUDahzNENtyg8mt/y+6ktIkMspP7e6eD6msSopDH7UxscpxovnQpr/Wg/fHdqrJax3Yf6BUZGuotsGUUrIWKKV/d5GDvaVABzkxpQfYBV7hrh7KTx+TguDX8JOpa5ShNYVWYmqkiR1SbReBiWzYJd/csDqJVm05ORKSGRqKgAgPaKcZH6GzcW3SwNdm04wBCESzxoia4uKKTvlBA3YERAyQfz8AO2n0fP458gfYfGzG1Gc5NntyNIGFigkkweEQiqnS7KiSrQNzynMlf38GgHBNZEw/dcP0C5Ae87PTh4h8kvR0arMGd23T9Fiu10oSqKRS2M1AjJk1YuUpznZP+BGQGCdRD+/qRvuN9jBVNU73cDbIT+aPhwQQ1j/oDufOZ1SlN0DZrqQM0JcklRVErl8HGXImgpF1tvyj1doZBox6yz/cja/xMvOdUohYVeTipTsTIHd1c+UYTj8qFqkrJClMUjMu2anbFxyNsHlOFwlU0QlkdNouJIidA4n+zf8aA1+nMQ19oo3+CXqrk2XJGcrFk6YCNq3/5T80EJa/6A+KuHGFFzc5kRHQDOlifu6wXVxOQRdTGKijBMqz6BWO8tI5gY32e5P2/xu1GK8a58kq/8MRWES9IbhQeZHgz3Nk0Qr0rCmN4DYqUUtEoLLcapoTfWwMsHDAZNVEpdpMs+wikcyzJa2kGH44iRadW+Mwc3P8EQiCbYtTk60j4KDwo8tlLHbZT4sNmbBdl3GKQr6KWQPU4Q0beBm5oAEzPZcZqHkyVMeqmzu8JhKJKYyc/gdLjHXPmGaP+g0MaXTd4AE2wEcEH5UK3aZJiTd3ctGaajsT9IBEyGVUURJ1ugaCAZAfwKmaqJ7PIRUkTk7A5jc2ShF6ckLrg0G9BnKKbZoyrFE1u7GB4Qfyx4WMFSOyjkypt2Z2d/grR2AAklmzXYeXY30KoV+csHeVBQ37CpHIyV+gj6Fmiv80dNYOiYQULDobLcBZkD4RURHlUY3lsqOfh5+CMEMSBOOITc6TFJQ/0+x5EdPgXBkkHQUk4a5laaKTYPGVegalZ3HBWJw7i7h4PKj8lPtPSqKp4QWaIkscpAQtVLAnh4cuW/9I+l8JBFBZdrSJoFezDzFjLN8YZ/ccoRRDAttV9AHgh/d3mx3AGizwJUOoLBdXDfommY05t1JH2GrS97FHpY97IhAnZlEayWeL1l7DuBZUB0SujQXwZc4y0hcc5HafQ6uA37PhM6PfmrQ6nEiTNX9UXyiPzjcZhEf+4F+finR0+NjRY2Yc5wTfBE7gs4994VCFNc008lMvshJimfei/2Htj38AH4T31er3/2qYd+FzY8m30akQmd0BjNNQ9XFrEIjaFMe8c2AWSlsyw+yB8wJY+4PedX4h3pe5yRRlBQzwLofjeTpfLXj9/133/3w449/7bXxsPlxDu/F5CGb2w+AXK06uVDCHeDWwKMzAR8UxATjbkBrsmNJXciJEBEg9cw4t5aDlB9Y4CDfg8luWoK3c92C3/ff/cTAPeu3kPnh1KMRtyF5KAktk6lVq5MzJfbJSp53ZN1qwOZ4PIVLsyg/3ABjGga/UiPwmYgX+BHQHhWfUdT1RD7jC6Ftt3BQft//6tlgcsfDT3LkXiEjqUY6N1uhnzooxSozk5PVaq3W2MloRX3XKSAAOkM9i3eaJT/I7uuVQtWDXSvFPjdSKRA55g/B/xVGmHQC+mJefs//2IJd+PxQKo0yC1e2U4L+Zalag/lVJBr1b4SvBnxSVXCvetIYaiePOO6e9n3SDSLpxOct0wPd5ds+/z7fEl7o/JCYPQwSV6B0SW/utbrnAAeGh+Ao3ZCYnc/p1+soMf9EosoL/s/vOw37PgHLoi7rOz8MMg4nIUJiM53abHXL2kyAAyedrobraHYqpZ99EKd9p0lVCtF0GzpS2wlcv/kJIBXZNfUi2c146+WHIAeGfGtLhW6WcZZ+asbfByCZL0vttodD2eObFnoseuGdF1omj3D5kZxjqNZdx9sU/Voh5q9gDEjAjb83ci9Jq1JR4PxsMvrj6TbBje6ibsPv55//j/zyyy8//PDDs804hskvL/k7d+26llAC0kmX/QU48CdMVsU4y6A487VzL0aE7GYyonnPKWQim2KbFj2cVEpCXm55M+a8CO/ml7/8xayfj89/daX77+2r8dXUtMdyUPfkaQUnqG5JT9emNi/7bXp6r8hNtwxvtAeRSOhd3Bo8SpDjTz851RgmP5LrZJedy6anlyv8jhfH3qZh6FPJaayl3VWvptXiGS9uarVaysEv4AudiKIU9Vw334XkOBf16h+fxfnbuANgj/kJivOjGR3dKdFmH+/ruWvu/41rhO7vIyJ6snc4lsM3zP0KTaNEL087FXp5em8PKD/x15JhY4dEUgjM/vmA3vHTNHrTuSmXJuA/A27DPWwhmzQk3dvwM18rZLPSZt4d2Zp8VZjDKLxAw3vqgpl9xzDAOfod+WN37N9O6SG/JnfLbpndsAnYb1o6SYK2N2pMQtOXM7reNYQmiNuj93CDaVNq7trVkcZPhYxav90z3kv9WfelNcw6YtFoQXg68P5NJ7zs0RLTdecYujZiC+4aVZznN1ZGR99mvx11PpT+VfO7Yn9qLnMdChxK0Oie6AY6MCa4OUtwzX4lyfrtsvPjg/f9p30x01O1OVNwbX/mjP7sJd3Ie7b5tfbUFjZ6+61GHjmD/CzBvXitM8H5bXfv1l2ziDlD/ExPtQX3FL8UPDr6xvjZ4ffEntqK4DL9Ib3Tzc/hqT0DZ9vur9GJTyW/HnpqC4NicPyU8fOCC4Vbw6AYfP6FU8Gv4anHAs620W4v1W9QHiMdThsGxvoNzLJj9tSe2WBwsztKJwecaf0lRztK106U4DzWL25hFyPHZf0Ad/VkhbiWdnzgUlaI6/eQe2qhcwt92tBfCw8cCu6EFSNPYL0Hp7Hq97QKzmO943YSq9+ntx6AC62jdBKsB4I7m+BMe2Jwp6mIewrrDlwfOkoDbh0KLpI6qRP8kK0NuTOZU7uxptxoR2kIrp150J326VbP7f8Bym0RwLS/g0AAAAAASUVORK5CYII=',width=400,height=400)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcS2knCf3Gb6Crim-f2wQDU_f9oLLjMzKhGqvaJUl6AgJJED1YGm',width=400,height=400)
nRowsRead = 1000 # specify 'None' if want to read whole file

# Discriminant-analysis-churn-dataset.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows

df = pd.read_csv('/kaggle/input/churn-dataset/Discriminant-analysis-churn-dataset.csv', delimiter=',', nrows = nRowsRead)

df.dataframeName = 'Discriminant-analysis-churn-dataset.csv'

nRow, nCol = df.shape

print(f'There are {nRow} rows and {nCol} columns')
df.head()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcR1B-L8Si-a1ysL6f1126m1ckjj4LvVMLQaSs1ovE10ZPYQFiUf',width=400,height=400)
df.dtypes
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'data:image/png;base64,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',width=400,height=400)
df.describe()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQs_tZFSgUbWXLjgWfyTgO9cP1cGim73WCyzZxT_UMA4711XiiD',width=400,height=400)
# Correlation matrix

def plotCorrelationMatrix(df, graphWidth):

    filename = df.dataframeName

    df = df.dropna('columns') # drop columns with NaN

    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values

    if df.shape[1] < 2:

        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')

        return

    corr = df.corr()

    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')

    corrMat = plt.matshow(corr, fignum = 1)

    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)

    plt.yticks(range(len(corr.columns)), corr.columns)

    plt.gca().xaxis.tick_bottom()

    plt.colorbar(corrMat)

    plt.title(f'Correlation Matrix for {filename}', fontsize=15)

    plt.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcSp20-glZO16U8O4e6X_BqdVSRvgO6c0p3627Q4ldIIoGC5kGdC',width=400,height=400)
plotCorrelationMatrix(df, 8)
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcRKJvvcA-scVt7aUW33iz0ge_j3zMddpCsaAj1_-WkUjWGik2yp',width=400,height=400)
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcRYVp4jZZcFSYRpLfjSwdxkU2BdExjUqNsEwVKqp2GHCgRXPr26',width=400,height=400)
sns.distplot(df["churn"].apply(lambda x: x**4))

plt.show()
#codes from PSVishnu @psvishnu

num = df.select_dtypes ( include = "number" )
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcR4VjrKNc2t2IPEhtc_8wY1LwRivlUU48UNw_QpvKK5aIF76bKp',width=400,height=400)
counter = 1

plt.figure(figsize=(15,18))

for col in num.columns:

    if np.abs(df [col].skew ( )) > 1 and df[col].nunique() < 10:

        plt.subplot(5,3,counter)

        counter += 1

        df [col].value_counts().plot.bar()

        plt.xticks(rotation = 45)

        plt.title(f'{col}\n(skewness {round(df [ col ].skew ( ),2)})')



plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)

plt.show ( )
%%time

# > 10 sec

counter = 1

truly_conti = []

plt.figure(figsize=(18,40))

for i in num.columns:

    if np.abs(df [ i ].skew ( )) > 1 and df[i].nunique() > 10:

        plt.subplot(20,3,counter)

        counter += 1

        truly_conti.append(i)

        sns.distplot ( df [ i ] )

        plt.title(f'{i} (skewness {round(df [ i ].skew ( ),1)})')

        plt.xticks(rotation = 45)

plt.tight_layout()

plt.show ( )
plt.figure(figsize=(18,3))

sns.boxenplot(data=df.loc[:,truly_conti[0]],orient='h')

plt.title(truly_conti[0])

plt.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcT68vGzkI1qqte5Au8VHmM2pjdxDoHhgqK_g4KjLc2Ds_zT0p2O',width=400,height=400)
churn = [

    'account_length','number_vmail_messages','total_day_charge','total_eve_charge','total_night_charge','total_intl_charge', 'number_customer_service_calls', 'churn'

]
sns.pairplot(data=df,diag_kind='kde',vars=churn,hue='churn')

plt.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcSOUIbO4aevg5kmsDcENS-4b1uyT_FrfAaUpp-2OxobBUarhByZ',width=400,height=400)