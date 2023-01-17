# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
# print(os.listdir("../input"))
from IPython.display import IFrame

# Any results you write to the current directory are saved as output.
IFrame(width="800" ,height="600", src="https://app.powerbi.com/view?r=eyJrIjoiMjk5OGE1ZDctODgwYi00NzU3LTg4YmYtOWVlNGVhZDYyYzUwIiwidCI6IjFlNzI1MWY2LTk4MjItNDRjZi1hYjQyLWJkZmQ5ODJhMTFlMSIsImMiOjEwfQ%3D%3D")
IFrame(width="800" ,height="600", src="https://app.powerbi.com/view?r=eyJrIjoiYzBhM2U4MjMtYTJhNy00MGMwLWIyYTYtYjZjYzM1Mjk1MjI3IiwidCI6IjFlNzI1MWY2LTk4MjItNDRjZi1hYjQyLWJkZmQ5ODJhMTFlMSIsImMiOjEwfQ%3D%3D")
IFrame(width="800" ,height="600", src="https://app.powerbi.com/view?r=eyJrIjoiNDhlMmVhMzMtMGNhZS00YzRmLWI4YzktZWI5NzA5ZjcxZWE4IiwidCI6IjFlNzI1MWY2LTk4MjItNDRjZi1hYjQyLWJkZmQ5ODJhMTFlMSIsImMiOjEwfQ%3D%3D")
IFrame(width="800" ,height="600", src="https://app.powerbi.com/view?r=eyJrIjoiZmU0MjdkOGUtN2VkYy00Nzc2LTgyNWYtNzViMzMwMjU5ZWM3IiwidCI6IjFlNzI1MWY2LTk4MjItNDRjZi1hYjQyLWJkZmQ5ODJhMTFlMSIsImMiOjEwfQ%3D%3D")
IFrame(width="800" ,height="600", src="https://app.powerbi.com/view?r=eyJrIjoiYTZlZmRmYjItNDYxYy00NGUwLTk2ZTItYTA1ZjUyZmFhYjllIiwidCI6IjFlNzI1MWY2LTk4MjItNDRjZi1hYjQyLWJkZmQ5ODJhMTFlMSIsImMiOjEwfQ%3D%3D")
IFrame(width="800" ,height="600", src="https://app.powerbi.com/view?r=eyJrIjoiNDY1MjQzMWYtNTQxNS00ZTczLTg2NmUtZjNhMDY2YWYyODU5IiwidCI6IjFlNzI1MWY2LTk4MjItNDRjZi1hYjQyLWJkZmQ5ODJhMTFlMSIsImMiOjEwfQ%3D%3D")
IFrame(width="800" ,height="600", src="https://app.powerbi.com/view?r=eyJrIjoiOGZhZTgzNGYtMDg3Yy00YWI5LWEzZDYtNTRhMTgxNWUxMTRjIiwidCI6IjFlNzI1MWY2LTk4MjItNDRjZi1hYjQyLWJkZmQ5ODJhMTFlMSIsImMiOjEwfQ%3D%3D")
