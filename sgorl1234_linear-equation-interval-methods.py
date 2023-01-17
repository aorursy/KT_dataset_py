# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!pip install -q pyinterval
from interval import interval
from interval import imath
import pandas as pd
import numpy as np
import math
pd.set_option('display.max_rows', 999)
def calculate_width(interv):
    return sum([x.sup - x.inf for x in interv])

def calculate_midpoint(interv):
    return (interv[0].sup + interv[0].inf) / 2

def print_interval(interv):
    return f"[{interv[0].inf:9.7f}, {interv[0].sup:9.7f}]"

def print_results(interv, interv_ext, root, tbl):
    tbl.loc[len(tbl)] = [interv, interv_ext,root]
def dichotomy_method(f, start, end, e, tbl):
    x = interval[start, end]
    width = calculate_width(x)
    mid = calculate_midpoint(x)
    Fx = f(x)
    root_exists = 0 in Fx and width < e

    if 0 not in Fx or root_exists :
        print_results(print_interval(x), print_interval(Fx), 'Root' if root_exists else '', tbl)
        return   
        
    tbl.loc[len(tbl)] = [print_interval(x), print_interval(Fx), '']
    dichotomy_method(f, start, mid, e, tbl)
    dichotomy_method(f, mid, end, e, tbl)
    
    
    
func = lambda x: x*x - 7*x + 10
e = 1e-2
a = 1.9
b = 5.3


dichotomy_method_result = pd.DataFrame(columns=['Interval', 'Interval extension',' '])
dichotomy_method(func, a, b, e, dichotomy_method_result)
dichotomy_method_result
def moore_method(f, df, start, end, e, tbl):
    x = interval([start, end])
    width = calculate_width(x)
    Fx = f(x)
    
    if 0 not in Fx:
        print_results(print_interval(x), print_interval(Fx), '', tbl)
        return
        
    if width < e:
        print_results(print_interval(x), print_interval(Fx), 'Root', tbl)
        return
        
    x_middle = calculate_midpoint(x)
    dFx = df(x)
    if 0 in dFx:
        print_results(print_interval(x), print_interval(Fx), '', tbl)
        moore_method(f, df, start, x_middle, e, tbl)
        moore_method(f, df, x_middle, end, e, tbl)
        return
        
    f_middle = f(x_middle)
    U = x_middle - f_middle / dFx
    x_next = U & x
    if not x_next:
        print_results(print_interval(x), print_interval(Fx), '', tbl)
        return

    print_results(print_interval(x), print_interval(Fx), '', tbl)
    moore_method(f, df, x_next[0].inf, x_next[0].sup, e, tbl)
    
func = lambda x: x*x - 7*x + 10
dfunc = lambda x: 2*x - 7
e = 1e-6
a = 1.9
b = 5.3

moore_method_result = pd.DataFrame(columns=['Interval', 'Interval extension', ' '])
moore_method(func, dfunc, a, b, e, moore_method_result)
moore_method_result
def hansen_method(func, df, start, end, e, tbl):
    x = interval([start, end])
    width = calculate_width(x)
    Fx = func(x)

    if 0 not in Fx:
        print_results(print_interval(x), print_interval(Fx), '', tbl)
        return

    if width < e:
        print_results(print_interval(x), print_interval(Fx), 'Root', tbl)
        return
        
    x_middle = calculate_midpoint(x)
    f_middle = func(x_middle)
    dFx = df(x)
    if f_middle == 0.0:
        print_results(print_interval(x), print_interval(Fx), 'Root?', tbl)
        hansen_method(func, df, start, x_middle, tbl)
        hansen_method(func, df, x_middle, end, tbl)
        return

    U = x_middle - f_middle / dFx
    x_next = U & x
    if not x_next:
        print_results(print_interval(x), print_interval(Fx), '', tbl)
        return

    print_results(print_interval(x), print_interval(Fx), '', tbl)
    [hansen_method(func, df, x.inf, x.sup, e, tbl) for x in x_next]
    
    
func = lambda x: x*x - 7*x + 10
dfunc = lambda x: 2*x - 7
e = 1e-6
a = 1.9
b = 5.3

hansen_method_result = pd.DataFrame(columns=['Interval', 'Interval extension', ' '])
hansen_method(func, dfunc, a, b, e, hansen_method_result)
hansen_method_result
def krawczyk_method(f, df, start, end, e, tbl):
    x = interval([start, end])
    width = calculate_width(x)
    Fx = f(x)

    if 0 not in Fx:
        print_results(print_interval(x), print_interval(Fx), '', tbl)
        return

    if width < e:
        print_results(print_interval(x), print_interval(Fx),'Root', tbl)
        return

    x_middle = calculate_midpoint(x)
    dFx = df(x)
    if 0 in dFx:
        print_results(print_interval(x), print_interval(Fx),  '', tbl)
        krawczyk_method(f, df, start, x_middle, e, tbl)
        krawczyk_method(f, df, x_middle, end, e, tbl)
        return

    dFx_middle = df(x_middle)
    K = x_middle - f(x_middle) / dFx_middle + (1 - dFx / dFx_middle)*(x - x_middle)
    x_next = K & x
    if not x_next:
        print_results(print_interval(x), print_interval(Fx), '', tbl)
        return

    print_results(print_interval(x), print_interval(Fx), '', tbl)
    krawczyk_method(f, df, x_next[0].inf, x_next[0].sup, e, tbl)
    
func = lambda x: x*x - 7*x + 10
dfunc = lambda x: 2*x - 7
e = 1e-6
a = 1.9
b = 5.3

krawczyk_method_result = pd.DataFrame(columns=['Interval', 'Interval extension', ' '])
krawczyk_method(func, dfunc, a, b, e, krawczyk_method_result)
krawczyk_method_result
def calculate_determinant(m):
    return m[0][0]*m[1][1] - m[0][1]*m[1][0]

def calculate_width(interv):
    return sum([x.sup - x.inf for x in interv])

def calculate_midpoint(interv):
    return (interv[0].sup + interv[0].inf) / 2

def half_interval(x): # subinterval
    mid = calculate_midpoint(x)
    return (interval[x[0].inf, mid], interval[mid, x[0].sup])

def to_table(x, y, root, tbl):
    tbl.loc[len(tbl)] = [print_interval(x), print_interval(y), root]
def moore_method_for_system(f, df, intervals, e, tbl):
    x, y = intervals
    x_width = calculate_width(x)
    y_width = calculate_width(y)

    f_xy = f(x, y)
    if 0 not in f_xy[0] or 0 not in f_xy[1]:
        to_table(x, y, '', tbl)
        return

    if max(x_width, y_width) < e:
        to_table(x,  y,  'Root', tbl)
        return

    x_middle = calculate_midpoint(x)
    y_middle = calculate_midpoint(y)
    df_xy = df(x, y)
    df_xy_det = interval(calculate_determinant(df_xy))

    if 0 in df_xy_det:
        x_left, x_right = half_interval(x)
        y_left, y_right = half_interval(y)
        if x_width < e:
            to_table(x, y,'', tbl)
            moore_method_for_system(f, df, (x, y_left), e, tbl)
            moore_method_for_system(f, df, (x, y_right), e, tbl)
        elif y_width < e:
            to_table(x, y, '', tbl)
            moore_method_for_system(f, df, (x_left, y), e, tbl)
            moore_method_for_system(f, df, (x_right, y), e, tbl)
        else:
            to_table(x,  y,  '', tbl)
            moore_method_for_system(f, df, (x_left, y_left), e, tbl)
            moore_method_for_system(f, df, (x_left, y_right), e, tbl)
            moore_method_for_system(f, df, (x_right, y_left), e, tbl)
            moore_method_for_system(f, df, (x_right, y_right), e, tbl)
        return

    f1m, f2m = f(x_middle, y_middle)
    df1x, df1y, df2x, df2y = *df_xy[0], *df_xy[1]
    U_x = x_middle + ((-df2y/df_xy_det)*f1m + (df1y/df_xy_det)*f2m)
    U_y = y_middle + ((df2x/df_xy_det)*f1m + (-df1x/df_xy_det)*f2m)
    x_next = U_x & x
    y_next = U_y & y

    if not x_next or not y_next:
        to_table(x,  y,  '', tbl)
        return

    to_table(x,  y,  '', tbl)
    moore_method_for_system(f, df, (x_next, y_next), e, tbl)
    
    
    
func = lambda x, y: (x**2 - y, x - y + 2)
dfunc = lambda x, y: ((2*x, - 1), (1, -1))
initial_intervals = (interval[-2.2, 2.2], interval[0, 4.1])
e = 1e-6

moore_system_result = pd.DataFrame(columns=['x', 'y', ' '])
moore_method_for_system(func, dfunc, initial_intervals, e, moore_system_result)
moore_system_result
def hansen_method_for_system(f, df, intervals, e, tbl):
    x, y = intervals
    x_width = calculate_width(x)
    y_width = calculate_width(y)    

    f_xy = f(x, y)
    if 0 not in f_xy[0] or 0 not in f_xy[1]:
        to_table(x, y, '', tbl)
        return

    if max(x_width, y_width) < e:
        to_table(x, y, 'Root', tbl)
        return

    x_middle = calculate_midpoint(x)
    y_middle = calculate_midpoint(y)
    df_xy = df(x, y)
    df_xy_det = interval(calculate_determinant(df_xy))  
    f1m, f2m = f(x_middle, y_middle)

    if f1m == 0.0 and f2m == 0.0:
        to_table(x, y,  '', tbl)
        x_left, x_right = half_interval(x)
        y_left, y_right = half_interval(y)
        hansen_method_for_system(f, df, (x_left, y_left), e, tbl)
        hansen_method_for_system(f, df, (x_left, y_right), e, tbl)
        hansen_method_for_system(f, df, (x_right, y_left), e, tbl)
        hansen_method_for_system(f, df, (x_right, y_right), e, tbl)
        return

    df1x, df1y, df2x, df2y = *df_xy[0], *df_xy[1]
    U_x = x_middle + ((-df2y/df_xy_det)*f1m + (df1y/df_xy_det)*f2m)
    U_y = y_middle + ((df2x/df_xy_det)*f1m + (-df1x/df_xy_det)*f2m)
    x_next = U_x & x
    y_next = U_y & y

    if not x_next or not y_next:
        to_table(x, y, '', tbl)
        return

    if x == x_next and y == y_next:
        if(x_width > y_width):
            to_table(x, y, '', tbl)
            x_left, x_right = half_interval(x)
            hansen_method_for_system(f, df, (x_left, y), e, tbl)
            hansen_method_for_system(f, df, (x_right, y), e, tbl)
        else:
            to_table(x,  y, '', tbl)
            y_left, y_right = half_interval(y)
            hansen_method_for_system(f, df, (x, y_left), e, tbl)
            hansen_method_for_system(f, df, (x, y_right), e, tbl)
        return

    to_table(x, y, '', tbl)
    for xi in x_next:
        for yi in y_next:
            hansen_method_for_system(f, df, (interval(xi), interval(yi)), e, tbl)
            

func = lambda x, y: (x**2 - y, x - y + 2)
dfunc = lambda x, y: ((2*x, - 1), (1, -1))
initial_intervals = (interval[-2.2, 2.2], interval[0, 4.1])
e = 1e-6

hansen_system_result = pd.DataFrame(columns=['x', 'y', ' '])
hansen_method_for_system(func, dfunc, initial_intervals, e, hansen_system_result)
hansen_system_result
def krawczyk_method_for_system(f, df, intervals, e, tbl):
    x, y = intervals
    x_width = calculate_width(x)
    y_width = calculate_width(y)

    f_xy = f(x, y)
    if 0 not in f_xy[0] or 0 not in f_xy[1]:
        to_table(x, y, '', tbl)
        return

    if max(x_width, y_width) < e:
        to_table(x, y, 'Root', tbl)
        return

    x_middle = calculate_midpoint(x)
    y_middle = calculate_midpoint(y)
    df_xy = df(x, y)
    df_xy_det = interval(calculate_determinant(df_xy))

    if 0 in df_xy_det:
        x_left, x_right = half_interval(x)
        y_left, y_rigth = half_interval(y)
        if x_width < e:
            to_table(x, y, '', tbl)
            krawczyk_method_for_system(f, df, (x, y_left), e, tbl)
            krawczyk_method_for_system(f, df, (x, y_right), e, tbl)
        elif y_width < e:
            to_table(x, y, '', tbl)
            krawczyk_method_for_system(f, df, (x_left, y), e, tbl)
            krawczyk_method_for_system(f, df, (x_right, y), e, tbl)
        else:
            to_table(x,y, '', tbl)
            krawczyk_method_for_system(f, df, (x_left, y_left), e, tbl)
            krawczyk_method_for_system(f, df, (x_left, y_rigth), e, tbl)
            krawczyk_method_for_system(f, df, (x_right, y_left), e, tbl)
            krawczyk_method_for_system(f, df, (x_right, y_rigth), e, tbl)
        return

    f1m, f2m = f(x_middle, y_middle)
    df_mid = df(x_middle, y_middle)
    df1x,  df1y,  df2x,  df2y =  *df_xy[0], *df_xy[1]
    df1xm, df1ym, df2xm, df2ym = *df_mid[0], *df_mid[1]
    det = calculate_determinant(df_mid)
    retard_x = (1 + df2x*df1ym/det - df1x*df2ym/det)*(x-x_middle) \
                + ( df2y*df1ym/det - df1y*df2ym/det)*(y-y_middle)
    retard_y = (     - df2x*df1xm/det + df1x*df2xm/det)*(x-x_middle) \
                + (1 - df2y*df1xm/det + df1y*df2xm/det)*(y-y_middle)
    K_x = x_middle + ((-df2ym/det)*f1m + ( df1ym/det)*f2m ) + retard_x
    K_y = y_middle + (( df2xm/det)*f1m + (-df1xm/det)*f2m ) + retard_y
    x_next = K_x & x
    y_next = K_y & y
        
    if not x_next or not y_next:
        to_table(x,  y, '', tbl)
        return

    to_table(x, y,  '', tbl)
    krawczyk_method_for_system(f, df, (x_next, y_next), e, tbl)
    
    
func = lambda x, y: (x**2 - y, x - y + 2)
dfunc = lambda x, y: ((2*x, - 1), (1, -1))
initial_intervals = (interval[-2.2, 2.2], interval[0, 4.1])
e = 1e-6

krawczyk_system_result = pd.DataFrame(columns=['x', 'y', ' '])
krawczyk_method_for_system(func, dfunc, initial_intervals, e, krawczyk_system_result)
krawczyk_system_result