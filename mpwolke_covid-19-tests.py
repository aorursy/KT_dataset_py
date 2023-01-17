#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQ8PwwNxDTTayMcqljjRFnC4ojRrE9kthc1z97Q__mEyIHcnEnT',width=400,height=400)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import feature_extraction, linear_model, model_selection, preprocessing

import plotly.graph_objs as go

import plotly.offline as py

import plotly.express as px



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/covid19testscountry/covid-19-tests-country.csv")

df.head().style.background_gradient(cmap='twilight_shifted')
plot_data = df.groupby(['Total COVID-19 tests'], as_index=False).Year.sum()



fig = px.line(plot_data, x='Total COVID-19 tests', y='Year')

fig.show()
plot_data = df.groupby(['Total COVID-19 tests'], as_index=False).Entity.sum()



fig = px.line(plot_data, x='Total COVID-19 tests', y='Entity')

fig.show()
fig = px.bar(df, x= "Year", y= "Total COVID-19 tests")

fig.show()
fig = px.bar(df, x= "Entity", y= "Total COVID-19 tests")

fig.show()
cnt_srs = df['Year'].value_counts().head()

trace = go.Bar(

    y=cnt_srs.index[::-1],

    x=cnt_srs.values[::-1],

    orientation = 'h',

    marker=dict(

        color=cnt_srs.values[::-1],

        colorscale = 'Purples',

        reversescale = True

    ),

)



layout = dict(

    title='COVID-19 tests by Age',

    )

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="Year")
cnt_srs = df['Entity'].value_counts().head()

trace = go.Bar(

    y=cnt_srs.index[::-1],

    x=cnt_srs.values[::-1],

    orientation = 'h',

    marker=dict(

        color=cnt_srs.values[::-1],

        colorscale = 'Greens',

        reversescale = True

    ),

)



layout = dict(

    title='COVID-19 tests by Entity',

    )

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="Entity")
# Grouping it by job title and country

plot_data = df.groupby(['Year', 'Total COVID-19 tests'], as_index=False).Entity.sum()



fig = px.bar(plot_data, x='Year', y='Entity', color='Total COVID-19 tests')

fig.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wCEAAkGBw8QDxAPEA8VFRUVEBUVFRUVFRUWFhUWFRUWFhUVGBYdKCkhGB4lHRUXIjEhJikxLjAwGB8zODMtNyguLisBCgoKDg0OGxAQGzcdICUtLS01NSs4MC0tLy0tLS0tLS0tLS43NS0tLTItNS0rLTUtLi0tLS0tLS0tLS0vLS0tLf/AABEIAJEBWwMBIgACEQEDEQH/xAAcAAEAAgMBAQEAAAAAAAAAAAAABAUBAgMHBgj/xABKEAACAQIEBAIHBAYGCAYDAAABAhEAAwQSITEFEyJBUWEGFBYyUlSSI3GBkRVCk6Gx4TNigqPS8CRDVWRyc9HxByVTsrTBNDWi/8QAGQEBAQEBAQEAAAAAAAAAAAAAAAECAwQF/8QALREBAAECBAQFBAIDAAAAAAAAAAECEQMSUdEUU5GhBBMhMYFBUmFxM7EiIzL/2gAMAwEAAhEDEQA/APZsZfuqyhLYYZXLMzEZcoEAQDJJPlsareDcfN7D4bFXba2rd/DWrqnOWIe7ly2vdEnrAEak9quMUTkeFLHKYAiTpsJgV8lguC4leH8GttbIuYN8ObtuVlhbsPZcKQYMF8411yeNBd8S4/as27dwA3A2KtYchZlHuXFt9Q7EZgcp1qW/E7CtkN0A5kUgzo1z3Fb4S0iAYJkV8xj+C4hziby2/f4rgsQtuRmNrDnDq7ntmItMQJmAvfQY43gcZfN3/Rcv+m4G6mQ2hntWbti47XGnM1wZbgy7QFid6C94FxG7eu4+3cyxYxnKQqCJQ4exdGaSZM3SJEbDSt8Bxy1dW+7TbWzee2xeQOggZpO0ztvUf0dwl23f4i9y2VF7GC7bJKnMgw1i1OhMdVptD5VTY7g+KaxirS2ST+k1xa9aqLqJfs3simZDEKwExDKO2tB9SOKWCDFwSLnLKwc4uZc+TJ72bKc0RtrtUa5x20MRh8OoLc5LzB1BKjksisNtTL/hBmq/H8It3lkYa7bL31uNdW5F+26WmVLynMdR0pGshiCCN+GEwWNF3h+IvWw727OLt3cmRTmuvbNt2AMAsLctlmGbwoPorGPsu2VLgJOeN9eW2V8p2bKSAY2Jiq3H8VveueqWcgK2UvO1wPlKtdysARtCLcg/EV8CDE4Lw2/ZxQZFZbD27r3bTlWW1fZ1INhpLBXlyV90QNjoeXH+D38RiMaqKVW/wlsMl2VyrcY3TqJzR1jWKC/t8Ww7AsLoIFvmTrrb/wDUHxL/AFhpTA8Vw98kWbyuRbS50mei5OR/MHK2vkfCqK/w+/cvWLwslBa4dftMsrma5e5MW1gwQOUSTMarE6xa+i2EazgsJauW8ly3hrNpx0zNtApEiQRMx99BLXiNkuLYuLmLlAJ3dQWZAdiwAJKjUQfCtbPFMO5AW6DKM6xPWikBmT4wCRqs7jxFUXBuF3UwzYO5Y61uYrJiOgrF97rLdUzmDEXYIga5u2p04Xwy+o4UGtZPU8My3IKwWFlbIt2/iUkFpMe6swdAFzZ9IMG4zLibZHLW4CG0KM2RWU/rSxC6dyB3rrd4xhkRrj3lVVuLbYt05XaMqMDqpOZYB+IeIr5nA8DxA4Zw+0bOW9hruGa5bJSXFpwzgMCQdyw11IG1Z4zwjEXfXbiWiedicCUSVBKYa4j3LhkgCQCImekeNB9MOK4eLhN1Ry2VXzdJUvGQEHXqzCPGRE1j9L4eUHNEvdNpQZDG4ozG3l3DRrG8a7VQcQwWK9Yx1xMMHW6cEFzctgVtOecyqxjOoaVzQJUbxryPBcS1riEW2W766uKwpdk6nt2rOQEqTllrbIw2hj2oL7E8T/0jD2ku2gGu3EdXD52KWmfLa7ZhAYz+qDWD6SYEAt6zbgLcYmdALJi7PhkPveHeofEsBeOI4Y62ywtYi7cvMCsLzMPetyJIJ67o2G1QcBwa8eGY7DtZyXbr8QNvMU1GJu3Xt9QJiQ6T93lQfRvxTDqzK15QVs85pMRa73CT+r511w+KS4SFaSACQQQRmmJB1ExXy93A3rl5mfBlrbcJawUdkGdy0taYZtARpO2/kTZ+jWCv2DftuzvZUoLDXSDey5SWRn3dVJhWbq1MzoSGnpFx7ki3yWRm9ewti4CrMAL963bdcwICuFuZoM9tKlYTigzYnm3bRVMQtpAmfOuZEIt3Ad7hZiQFGzLXzl3hGKXDHDCwzsvGVxWcMkPZOPXFFpJHUFJXKdZXTSKktwa5cuY8XbFwW72OtXUdHC3FW3hrSLdQq0qy3LW3cHYzFB9IeJWYB5gOZmUAAliyTnUKNSVymRGkGa5vxnChBc56FTaF0MGBXlHa6SNAn9Y6edfNYThuOW7g8RiFa7y0xll+WVt3Ct29baxfYKVUsUsjOB3fbeNPSDg198PisLh8GER+GPas5Dbzcx+bNq45MhRNsgLpLNJNB9dcx9lWyFwCCoO8AuYQE7AkkQDvNajiNk3Ba5i5ixQDxdRmZAdiwAJK7wD4V8/i+G4g4hMTYR7bm7h1uKxU2b1hcmdriyctxJfKywZVRqNs+j3CGtAWb+GZmtYy/et3c82zzbl11ugTIfLdKkZdy3YzQWeO4m6Y7DYcZRbuWMRcckHMDZNkLBmAPtGmQdhtWuE4vcvYUYy1ZzI1s3LSEkO9uJRttCwghY7jWdKh8YtB+KYJDs2Bxyn7i2FB/jXXgtrFYbh9vC8rNesWBZRhl5dzlrlS5M6AgAlTqNR50EyzxNr13EW7CgiwwtszkgNdKK5QQNgrrLeJiDBqvwnpTzLmEUWcqX3v2WZn6rWJw5bNYKxDTy7kNP6nmK78Lwl3C3sb0F7d6/6xbKRIZraK9tgSIOZJB2hu0VXN6OXl4dcRY9Z9auY5AG6RiGvtiFt5vh15ZPcE+NBY8b43dwyPcFlGUYnD2F+0ZSTfuW7ZY9JjKbg8Zg7V1t8VuvjMThFtJ9jYsXQxdhn5xvKFIynLHJOuu4qL6UcPvXMJatWkLuMXhbrAFRpbxVu9dPUR2VoH3VthrF5OJYzEGy5t3MLhbaEFOprTYlnEE6f0q6mO9BZcH4kmJtc1QVh3tup3S5acpcU+MMp17iD3rtex1lLlu091Fe4SLaFgGcgFjlXcwAT+FVXCuBvbslWvujviL19+UVjNeuM+SWUyFBAmBMT3rnxnAXjf4YUV7i2MU1y45KSFOHv2gTtJzXF2G00FsnEbLPy1urmJdR4Fk99QdmK9wDIgzUX0Z4g+JwqXrgUMWuA5QQvRddBoSeyjvVBh+E3lu4DCqUdcLjL2Ie4pbMLbJfCKwiFcm+BEmQGbTarT0E//AF9r/mX/AP5F2g859JPS/i1vH4mzb4hasW/W2sYdblkNnZVtkrmCNlUc1JZtOoeBijT079JS5th2zggFeRZkEoXE6R7qlp8ATUD024tyuK41WsW7vLxrXbXMz/Z3CloE9LDOp5aSrSOkec12L4viuVcDWA2e1bF2+BcIbNhzat5m9xWyXgIAEsK9cUxaPRxmX02F9N/SC5a5yYy3l9XvXj9ja/1D5Xt+77+xA8GFMR6ZekyNlFwvqqymHtEFmUEqOmTBlSdsykdq+cwTY3D4a1y+UyretY3lhpu5WCoA6bm03RmAntMV0tcS4gCbvqRNxrj2Sxt3c5zXGxRs8udCGuZpiYAHarljSC76G76Y+kihz6ysJhmvM3Is5VK23ucsnLq3RGniK9zwFwtZtMxktbUn7yoJr8r3PSC4LAstZQ8uxctIxzhkFyyLNwwDBJVV3GhXzr9S8L/oLP8Ayk/9orji02s3RN0i5MGN4MffVYGxAiGVjlyxK6lQeo7dRIgx2BirNwYMGDGh8POoeH4cEZSHJCmQCB3Ug6jzJO25NcWwNiOswPcbINILAnLOs+H5VjPiZPSD70ba/DOuxG/ef3YXh7CPtTpBGh7GYOuo8e5jeg4cYjmmYIJM+AEjXTb95oBOJgaD3h2WYzLPeNswrN5sRnJVZGw92I0M7yfD8fxrZcEQrjme8F7TEb7nv4dqWMCVZW5haJmZ1JnXfzA/sjwoNbFy/qCoMWzrKyX0gaHSZO/ltWOZiY9xZ07jw1nXtH/9DwNarwxgD9qde4BBnKFkGdzGp7+Vbvw9jP2kCDooIAJVV017ZZH3mg5D1oADL8JJlSdxIEnyO/jXbCm/mGcCIkxBhtZG+nbx71gcO1M3CRO2u2bNl32jSP8AtWq8NI/1p79jqTJJOuszr4wNomg1tNiQq6BtBrpvCTJnxL7eArob15VbPlBlQuwmWIYCTvlgjzo/DyVCi4RCkSAREzqNdN/PTwre5gSQIbYXBquYZbhmIntAA/hVj8pKORiQc8jb3SQFkqgjfxzf5itjcvkwo0LamVmAADpJg5gRp41t+jdwbhIylQNdJJIbfUif3CtU4XGouHt2+/Mwj9YsS0+J2NW1OqXnRvevXgzQFyhtyR3jQ66aE+e2lanEXgRmCqDlABK5joc0dUbx+BrV+EyuXmGCBMgnUBBO+/T38a3bhxkkXNw4IK5tHJJjXeIH4CranUvOjew+IkZ1X3T4aHSO5kTSw18sMygLpO07NOxPcL+B/LkeGNJPN1OYCVmAxTTf+pv510bAsSCbh/VncSFI0OuxA18/yqWjUvOjjGKULqGOQgCV1Y5TLbbajTwJ76dHOI7wuoMiIAD6zr8IH5mtbHD3yQ7iZ03IPukztM5Y7bneTWzcOYz9qwmdp2Mnx3nLr4LHelo1LzoXjeJSJ/ousArAJZSY1mSA4B2/jWbnN5QEPmDRIKggRIYknWJH3keFDw6RcBec4I282InXX3vLatX4YSCDc3zzIJ0YRtMaePlFLRqXnRoBfMRm97U5kKhgeoxPunYDt1aTFScMXVerTXQOwnKFGaSJnWT9xFR73DXynLc1LE6yBBYHL30EV2vYHMFlhorA6SeoRAJOg8taWjUvOjW+LrFzbJgqmUyuWZbNBBkbqZ8vwrS8t+dJ3EkFVBQHqAE6Md58PCthw06nmGcxbQEAEsGMCdtx/aNE4cQNbhOvcGI10AnSJ0+6lo1LzoyBd+zJBbpXZgIM9Rb4tI2nY+VcksYkqAzGRajRoJbpMsR394eGk99J2FsctcoMjt5aAfz/ABrtWZacuShKsUGZZylgCw8YPaYFdaUoFKUoFKUoFKUoMFQRBGhrWzaVFCooUDYKAAO+wrelB+Yf/EWw54vjyEYj1g6hSf1VqPe4zea1yvV2A5DWt2/WtYW3miP91Bj+t5a/onG+h/D71x7tzD5ncyxz3BJ8YBiuHsLwv5X+8u/4q352JpHWXaMLw31qq6Ru8Aw/GChtOMHcLrh7VhibjZWtW2UsFUICjMFy5sxgEwJiOt3jucKrYF4X3QHAj/Rlw8lRbynRFMZcu4g9vevYXhfyv95d/wAVPYXhfyv95d/xVPOr+2OsteX4X7qukbvzTxZrl67iL3KZebcuXMsMcvMZmiY1iYr9YcM/oLP/ACk/9oqlPoJwv5X+8u/4q+itoFAUDQAAfcNqTiVV/wDUWc66MKn+OZn9xEf1MtqUpWXNUcRxrWr46sy8uRaQpnLAOSSpElTA1U6EeFczx2A0hDCOQyvKuyi0QimNT9oR/Z/K7rFBRW+KXs0nKQr3VYA7BcQLa9t8pn8vHTrjMfdS+y68sZDIAJmGIt+WcgDMfuEFgRcVmgpP008ErbV8qO5yMSCERWyLpq3VH5Hyra3xok2wVXqcAw4bRmyhgRI33EzpW1jjJysWt7KpkMgDFmcQMxGvT/Gt8PxhCYIPUwFuB7wOTz3GcE+WvY0Hnfpfx/F2sfiLdvFOiqygKGgDoU/xJqn9qMd87c+qvpvSf0txNjGX7KJZKoygFrctqinUzrvVX7c4z4MP+y/nXSPb2fBxqo8yr/bMes66/tW+1GO+dufVT2ox3ztz6qsvbnGfBh/2X86e3OM+DD/sv51fhzzxzZ6TurfajHfO3Pqp7UY75259VWXtzjPgw/7L+dPbnGfBh/2X86fBnjmz0ndW+1GO+dufVT2ox3ztz6qsvbnGfBh/2X86e3OM+DD/ALL+dPgzxzZ6TurfajHfO3Pqp7UY75259VWXtzjPgw/7L+dfVei/H2vYW5evWbbOMRy1CJlEZFck77DMfw01qT6fR0wqPNqy04s3+d3wftRjvnbn1U9qMd87c+qvVcPj7bi6eVbAVS6sTplV3QltOn+jJ79/CuTcUAW7OFUPbscwqTALBUd0zZf1VuW9Y1LQNjUzRo9fAYvNnvu8v9qMd87c+qntRjvnbn1V7X6vb+BfpFPV7fwL9IpmjQ4DF5s993intRjvnbn1U9qMd87c+qva/V7fwL9Ip6vb+BfpFM0aHAYvNnvu8U9qMd87c+qntRjvnbn1V7X6vb+BfpFPV7fwL9IpmjQ4DF5s993intRjvnbn1U9qMd87c+qva/V7fwL9Ip6vb+BfpFM0aHAYvNnvu8U9qMd87c+qntRjvnbn1V7X6vb+BfpFPV7fwL9IpmjQ4DF5s993intRjvnbn1U9qMd87c+qva/V7fwL9Ip6vb+BfpFM0aHAYvNnvu8U9qMd87c+qntRjvnbn1V7X6vb+BfpFPV7fwL9IpmjQ4DF5s993intRjvnbn1U9qMd87c+qva/V7fwL9Ip6vb+BfpFM0aHAYvNnvu8U9qMd87c+qntRjvnbn1V7X6vb+BfpFPV7fwL9IpmjQ4DF5s993intRjvnbn1VK4R6SY1sTh1bF3CGv2lILaEF1BH5V7D6vb+BfpFBYQa5F/IVM0aLHgcWJv5s991OnEbxykEMDfZINtlZgLbERr0DOBBM6EeM0t8SvcoksMwvWhJtsM1tuXzCFnSC1xZ7ZNZir2sVl9JSXOIYvMwGHYDn2wDCmLRZQdm6mMkyNFnXbW8pSg43cQFe2ke/mg+GUTUMcZtZC0wcrHKZjpVmgtEA5VJ+4GpeKwq3MslgVMqymCCQQfI6E6GozcHslMkHLHxH/0za3/4WP460G68Us92I62XVWGqMFY7e7JHVtqK1PFUi8VBIt2hcmCMw6/dnf3NxprWTwy3mD6yGZhsYzkMy6jYlQfHwNLfC7aq6dRD2xbIJ2QZoUeEZz50GqcWtkZ/9XJAedGyg5ioG4BET5Htv0/SljTr38VYRqw6tOnVW3j3T4Vi5wy0STB1MkA6SRBMeJG/3VCxnDrXOBa4FDI5IKz05ibsOTCg80A6TG0UGVxODJYtbAymAShByhVcsREqo5upOnV512t3sLzECwCjEKApChjNuZiP1WUGfEVq3DcOZBuEllbNLiWRlRWB/qkIm356muqYSxIhgSxDDq3i410EePUWoPhPSjjOEt42+lzhtu6yss3DcILdCmYjTePwqq9oMD/si1+1P+Grb0n/AET67f5/rPMzLnyZMk5FiJ12iqv/AMj/AN8/u63D4mLNeer/ACp9502Y/T+B/wBkWv2p/wANY9oMD/si1+1P+Gtv/I/98/u6tOFcD4PiLb3VfEqqMFOYrvlzTCg6AAkntBq+iUU41c2pqpmfjZa8PwfDHsWL1zAKguW2cn3kXKHYgmQdkOsdwO4qYeDcMy2n9SUrcIEjKQhJCw3V4nWJiD2FS8NgcDZsn7QlMOjWWY3HhR7xUgEKT1jtMx3Ajpet4NGOcuvKDXOo3gkHLnMnpuSSJ3ksfE1i761ODRli9MX/AEgYXhHCrvLyYRSLmeDGgy6666SNQN43ip3slw75S3+Rqfa4baUqygypLA53MkqFJaT1dIA1mABFTKXlrycP7Y6KT2S4d8pb/I1NwfCcNZRrduyiozZisSCdNYP3D8qnVipdYw6KZvEWczh7czkWSZJgamIk+caUuYe205kUyADKgyBqAfEV0pRsVQBA0A2FKUoFKUoFKUoFKUoFKUoFKUoFKUoM0pSgUpSgUpSgVis1igVmlKBWruFBLEAASSdABW1R+IYc3LVy2DBZGUE7AkaTQdg4kiRIiR3E7fwNYS4p2YHUjQjcGCPwOlVN7h164zOxCFhGVLjx023CnMACep520gHetLvC75ki51EMM2dgYN7mZRoRqsrMaedBdM4EAkCTA8zBMD8Afyqtxlpbt0xiFDIuUKILDM9tmzazrkUCIIzeMVys8Nui5aZmkKwbquMSoy3FKgQA2rgzp4dhWMVwy6zPlhUNxXy8xoZlu23zbfZmFaQCQS0+dBpZ4GV6eaAARkOWWJFjlaknaM3T++u+H4IqGc8nMhmPhvXL25JOvNI37Vx/RmILhy4EXAwAdyFOS6rMAf8AjQwd8pGlb4bh19eUS5JV5bNdZliEDGMoJPS0a7kkzJFB8b6UcGwdzG33ucSS0zMpNs2ySvQoiZ12n8ajYD0Nw1/PyeJo2QAtFowoMwTLeRqV6UeiWJv4y/eR7IV2UjNcytoijURptVh6K+iOJw4xAvFBn5JUqzNraucwg7ETAEjbftW7+nu+TGBnx5irD9Lz6+v5/KE3/hwAQDjlBMkA29TGpjr1q99HvRdbFh7QxedblwPKKoDBRBQg5gyzv90HSQbFeEPltoXWEtcowHGdOWVgqGgAkz4iNDrp0wvCWW5buteZiqusEAgKxJyqT1eEkmTlWdqzeXuw/C4WHVmpi0/Lrc4ZIvjmvF7NmEJpKLb6TGkKg3mmJ4TbuhxcLksFBYMVIyqQIyx3Zj97fdFhSo9DCiAB4CNTJ/PvWaUoFKUoFKUoFKUoFYrNKBSlKBSlKDFZpSgUpSgUpSgUpSgUpSgUpSgUpSgUpSgUpSgUpULi7sthypIMrqNwMwDH8poNuKLca0yWx1NCzMQpMM09oWY84qus+toBbymA1tVIysuQXuoljBP2RA8ZUx41q3ELqFhb67Zboe5tIWTbDaSCYhtf1hrEV29fvlguVRmZgpyscoV3WW16pAB7UHALjDkOa5IR1mLUE5rcMVjp0zwNfd3117BcStxQoYrzZZvswHUkBiwAmY1kEfca5Lxi9lQsgzMVOUK2qsttiskjqGcidZyk5dDHSxjr5u2869LqswDlSWaAe5YwBOgEj8Q8+9NPR7F3sbirlvCu6sRlYLIP2ajT8RXoNzC35ugBirXVbVwGIFwtcUEHRSsKo33Bga1A4fiLxxfF15jwgs8sdThCbbE5E1Ez2A38asLmOvk2giPGTr6GDFyoPcQMoaYMSYAMgirMvN4fCppmquPrM9plG9RxUEdY6EUsLubOOnNGY6HTKDA0LmZYRf2c2Vc8ZsozZZy5o1idYmqnh2Ivm5aFwuVKX97ZEkXF5TM2UZTkzaED8dDVzUekpSlApSlApSlApSlApSlApSlApSlApSlApSlApSlApSlApSlApSlApSlApSlApSlAqqw+PuckYhihDLmFsAhgSQAuaTmYTliN/CrWoz8PsEsTZtksZYlFlo2nTWgiDi+45Le9kHw58wQqW23O4n3T5A74XHs117bpHWQpBHa3bYg/UdfKpJwVnX7JNQAeldQIgHxGg/IV0SwixlRRG0ACNAP4AD8KCkw3GrmZgy5uoqoCspnmXFXucwy22kgbjvOko8aAOtpgMwXWAcxVHgr2gP8AmD5VvxZ7dtUm1bYM5TrhVEg3DJg7lB23iuFq5hVZw1pAznK2VWcERbSC2UQuqCNu/jAbfpg8xxk6FyLOkhmvXbM+YJRfuE1twzibXMqlZYqrMRChQVQ6anNqx/zE6WMXhQtp2tKjcsMsW5gNEhSBrq/bxPnWzYzDE2wtoMwdQPs45ZLm3Mx0mbZH9n7qCKOLXxdvqFDBcYtpRAmDZtPl3HdnJbWANiNtF9IWL21U2nVr+TMhkRmtKQNZkG42sa5QYAOYSrWOyvfyWV//AC1ttBys5Nu0c5EdRAYz5JW68a6srWiIdUY5gQM7rbWCNG62ymNirbwJM0+3Va1msVmjRSlKBSlKBSlKBSlKBSlKBSlKBSsVmgUpSgUpSgUpSgUpSgUpSgUpSgUpSgUpWKDNKUoFKUoFVq8bs9OaUDe6XygEQxzTOg6DvrqPGrKoGH4Tat8vKIKPmmBLnK69R7/0hoF7F2jcRSCSrZlIIiYRJmfC+ND56bVoWwZl+Zb1bU8wQT0NG/8AVQ/l41HscLsAW8t0mDkXUatbFoEfePVtf7VYTgTZUDXf9TynIA93IqQoiOzGT4+GlB0tYXB5gNA0wqm5qeWcgIWdf6IfTr3rolnCFldXWS+kXTDMHNzYGGIa4T/a+6sjg1vrGZoZgTt2vXL0fncI+4ClnhCrkIcyrA5gArMAFGUkbg5BI7+UCA5WMThlu3QRDnFMOrWbgsIWZfAZIH5+OrBNgptLbtKDJyDlkZSFV/CF6cpH4eGmiYTDG87NcljiZgnL1BbDZB8Wtu20j7vGu2Es4X7HlXFgPcZAjLDsQc5ge97xOm00SFnWaUopSlKBSlKBSlKBSlKBSlKBSlKBSlKBSlKBSlKBSlKBSlKBSlKBSlKBSlKBSlKBSlKBSoGNuXw55a5gFBA6crGWzhidRpliO579uN7EYvUrZEAnpkZiFCsNzoW6l8jBrUUi1pVUb+LCxygSo1MgZiGUQNdJEme23atefi4INv8AXHUMshTqSF8oiDPveVMqXc7vAyUhbhDZ77SWcqOaLuWFmARzBt4Hxrf9F3GY3GZc2dWABJyxcViAfNQRMd63GJxQ05WgJ1gSRD5TAO/uTHnptW74vEaEWP1JOswQskabyZGnhPemUurcJwm+bYUgKQQSzZS2YBYYRMRB1mSWmR3mpwtla1B0lmvEfrwwZJO8yAP+GR4VYYO67orMsEgyNRGvh/P865W7l4DqUbD+Y00/7+U1lUYcKY3HcsBN83AVBz5ctqVDdpNvUQQRW1jhjK6ObgJFxnbo3zJkgGekd+/lAAFSBeu6fZjYk/f2H/1W3MuAN0z8PadToR22H50Eis1GFy7I6P59JMT2gwK1F278Pc6x21y6T935UEusVw5lyR07hZ8idxP+dvOsNcuz7n60fhO5/Cgk0qKl67llkg9Pid4BOm8Sfy861fEXQJNv8N418e/86CZSovNu/AO3j3H/AF/h5il2+4iE3AnyPh/H+dBKpUQXbu+Tx3/doK2L3YEAbmdNuoR3+GaCTSowu3Ms8vXNtPb7/wDO9YN278A0/wCo0/LWaCVSoyvcjUTsdo/W1H5VocRdkdG8+OwAg/v28qCZSowu3NSV2UxAOp0/ztWpu3dOgDUeek6n8u3n5UEulREuXu6Aa/kNf+1drLsfeWD+7YUHWlK4O7SYHf8AdA189ZoO9Kjcy58A/wAzP8P31kXLnwjY6ecHSfvigkUqM1y5rC9zH/0fx/dQvcKkxB021O/VE+AoJNKi5rmV95gxMRt0nzJ8Kxba7Bnw00197/oJ/tdoq2EulR87kbRqJjXtJjfxA/OtWuXdYUbGPv1j/P8A1qCVSo63Lk+5pI18p1/drWGa5rA8dxoInLHj2/Ogk0qPmeDp3HbWJ1I7HSteZc2y/cfKe/YGNaCVSuCXHkSv3/l/1rvQKUpQKUpQKUpQKUpQKUpQKUpQKUpQKUpQKUpQKUpQKUpQKUpQKUpQKUpQKUpQKUpQKUpQKUpQKUpQKUpQYrNKUClKUH//2Q==',width=400,height=400)