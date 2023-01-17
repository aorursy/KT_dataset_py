from IPython.display import display

from ipywidgets import widgets, Layout, Text, Label, HTMLMath, VBox, HBox

from sympy import Symbol, simplify

import timeit

from fractions import Fraction
def det_time():

    setup_code = '''

from __main__ import cal_det

from sympy import Symbol, simplify

    '''

    test_code = '''

cal_det()

'''

    times = timeit.repeat(setup=setup_code, stmt=test_code, repeat=1, number=1)

    return min(times)





def cal_det():

    global matrix, m

    m = [None] * 9

    for i in range(9):

        m[i] = f'{items[i].value}'

    matrix = m

    sym = list(set([i for j in m for i in j if i.isalpha()]))

    if len(sym) > 0:

        for i in sym:

            exec(f"{i} = Symbol('{i}')")

    else:

        m = [float(i) for i in matrix]

        return (m[0] * (m[4] * m[8] - m[5] * m[7])) + (m[1] * (m[5] * m[6] - m[3] * m[8])) + (m[2] * (m[3] * m[7] - m[4] * m[6]))

    deter = f'({m[0]} * ({m[4]} * {m[8]} - {m[5]} * {m[7]})) + ({m[1]} * ({m[5]} * {m[6]} - {m[3]} * {m[8]})) + ({m[2]} * ({m[3]} * {m[7]} - {m[4]} * {m[6]}))'

    return f'{simplify(deter)}'.replace('**', '^').replace('*', '')





def det(sender):

    # try:

    det1 = cal_det()

    if str(det1).endswith('.0'):

        det2 = int(det1)

    else:

        try:

            det2 = round(det1,5)

        except: 

            det2 = det1

    x = det_time()

    plc1 = det2

    plc2 = f'{x:.13f}'

    answer.value = r"<font color = 'black'><font size='3px'> $$Determinant: {} $$ ".format(

        plc1)

    time1.value = r"<font color = 'black'><font size='3px'>Time taken: {} seconds ".format(

        plc2)

#     except:

#         answer.value = r"<font color = 'white'><font size='3px'>Determinant: N/A"

#         time1.value = r"<font color = 'white'><font size='3px'>Time taken: N/A"





dic1 = {'answer': '$$Determinant:$$', 'time1': 'Time taken:'}



items = [Text(layout=Layout(width='60px', height='30px'))

         for w in range(1, 10)]

lst = [HBox(items[:3]), HBox(items[3:6]), HBox(items[6:9])]

display(VBox(lst))

for i in range(9):

    exec(f'items[{i}].on_submit(det)')



for j in dic1:

    exec(

        f"""{j}=HTMLMath(value=r"<font color = 'black'><font size='3px'>{dic1[j]}")""")

    exec(f'display({j})')
def inv_time():

    setup_code = '''

from __main__ import cal_det, cal_inv

import timeit

    '''

    test_code = '''

cal_det()

cal_inv()

'''

    times = timeit.repeat(setup=setup_code, stmt=test_code, repeat=1, number=1)

    return min(times)





def cal_inv():

    global matrix, m

    x1 = matrix



    m = [None for i in range(9)]

    if all([i.replace('-', '').replace('.', '').isdigit() for i in x1]):

        x = [float(i) for i in x1]

        m[0] = x[4] * x[8] - x[5] * x[7]

        m[1] = x[2] * x[7] - x[1] * x[8]

        m[2] = x[1] * x[5] - x[2] * x[4]

        m[3] = x[5] * x[6] - x[3] * x[8]

        m[4] = x[0] * x[8] - x[2] * x[6]

        m[5] = x[2] * x[3] - x[0] * x[5]

        m[6] = x[3] * x[7] - x[4] * x[6]

        m[7] = x[1] * x[6] - x[0] * x[7]

        m[8] = x[0] * x[4] - x[1] * x[3]

    else:

        x = x1

        e1 = f'{x[4]} * {x[8]} - {x[5]} * {x[7]}'

        e2 = f'{x[2]} * {x[7]} - {x[1]} * {x[8]}'

        e3 = f'{x[1]} * {x[5]} - {x[2]} * {x[4]}'

        e4 = f'{x[5]} * {x[6]} - {x[3]} * {x[8]}'

        e5 = f'{x[0]} * {x[8]} - {x[2]} * {x[6]}'

        e6 = f'{x[2]} * {x[3]} - {x[0]} * {x[5]}'

        e7 = f'{x[3]} * {x[7]} - {x[4]} * {x[6]}'

        e8 = f'{x[1]} * {x[6]} - {x[0]} * {x[7]}'

        e9 = f'{x[0]} * {x[4]} - {x[1]} * {x[3]}'

        sym = list(set([i for j in x1 for i in j if i.isalpha()]))

        for i in sym:

            exec(f"{i} = Symbol('{i}')")

        m[0] = f'{simplify(e1)}'.replace('**', '^').replace('*', '')

        m[1] = f"{simplify(e2)}".replace('**', '^').replace('*', '')

        m[2] = f"{simplify(e3)}".replace('**', '^').replace('*', '')

        m[3] = f"{simplify(e4)}".replace('**', '^').replace('*', '')

        m[4] = f"{simplify(e5)}".replace('**', '^').replace('*', '')

        m[5] = f"{simplify(e6)}".replace('**', '^').replace('*', '')

        m[6] = f"{simplify(e7)}".replace('**', '^').replace('*', '')

        m[7] = f"{simplify(e8)}".replace('**', '^').replace('*', '')

        m[8] = f"{simplify(e9)}".replace('**', '^').replace('*', '')

    return m





def inv(sender):

    global matrix, m

    # try:

    t1 = time()

    det1 = cal_det()

    if det1 == 0:

        answer.value = r"<font color = 'white'><font size='3px'>$$ Inverse: \ Determinant \ cannot \ be \ 0 $$"

        time1.value = r"<font color = 'white'><font size='3px'>Time taken: N/A"

    else:

        if str(det1).endswith('.0'):

            det2 = int(det1)

        else:

            try:

                det2 = round(det1,5)

            except: 

                det2 = det1

        m = cal_inv()

        k = []

        for i in m:

            if str(i).endswith('.0'):

                k.append(int(i))

            else:

                try:

                    k.append(round(i,5))

                except:

                    k.append(i)

        isneg = ''

        try:

            if det2<0:

                isneg = '-'

                det2 *= -1

        except:

            pass



        det2 = r'1{det}'.replace("det", f'{det2}')

        y = inv_time()

        plc1 = r'Inverse: {} \frac{} \begin{}  {} & {} & {} \\ {} & {} & {} \\ {} & {} & {} \end{}'.format(isneg,

            det2, "{pmatrix}", k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7], k[8], "{pmatrix}")

        plc2 = f'Time taken: {f"{y:.13f}"} seconds'

        answer.value = r"<font color = 'white'><font size='3px'> $$ {} $$ ".format(

            plc1)

        time1.value = r"<font color = 'white'><font size='3px'> {}  ".format(

            plc2)

#     except:

#         answer.value = r"<font color = 'white'><font size='3px'>Inverse: N/A"

#         time1.value = r"<font color = 'white'><font size='3px'>Time taken: N/A"





dic1 = {'answer': '$$Inverse:$$', 'time1': "Time taken:"}

items = [Text(layout=Layout(width='60px', height='30px'))

         for w in range(1, 10)]

lst = [HBox(items[:3]), HBox(items[3:6]), HBox(items[6:9])]

display(VBox(lst))

for i in range(9):

    exec(f'items[{i}].on_submit(inv)')

for j in dic1:

    exec(

        f"""{j}=HTMLMath(value=r"<font color = 'white'><font size='3px'>{dic1[j]}")""")

    exec(f'display({j})')
def solsys_time():

    setup_code = '''

from __main__ import cal_det, cal_inv, calsol

import timeit

    '''

    test_code = '''

cal_det()

cal_inv()

calsol()

'''

    times = timeit.repeat(setup=setup_code, stmt=test_code, repeat=1,

                          number=1)

    return min(times)





def calsol():

    global m, det1, a, b, c

    newdet = int(det1*1e6)

    a1 = ((a * m[0]) + (b * m[1]) + (c * m[2])) * 1e6

    a2 = ((a * m[3]) + (b * m[4]) + (c * m[5])) * 1e6

    a3 = ((a * m[6]) + (b * m[7]) + (c * m[8])) * 1e6

    x = Fraction(int(a1), newdet)

    y = Fraction(int(a2), newdet)

    z = Fraction(int(a3), newdet)

    return [x, y, z]





def solsys(sender):

    global matrix, m, det1, a, b, c

    det1 = cal_det()

    if det1 == 0:

        x1.value = r"<font color = 'white'><font size='3px'>x : Determinant cannot be 0"

        y1.value = r"<font color = 'white'><font size='3px'>y : N/A"

        z1.value = r"<font color = 'white'><font size='3px'>z : N/A"

        time1.value = r"<font color = 'white'><font size='3px'>Time taken: N/A"

    else:

        m = cal_inv()

        a = float(f'{items[9].value}')

        b = float(f'{items[10].value}')

        c = float(f'{items[11].value}')

        ans = calsol()

        t = solsys_time()

        for i in range(3):

            if int(ans[i]) != ans[i]:

                v = str(ans[i]).split('/')

                if ans[i] < 0:

                    v = [str(i).replace('-','') for i in v]

                    ans[i] = r'- \frac{num}{den}'.replace("num", f'{v[0]}').replace("den", f'{v[1]}')

                else:

                    ans[i] = r'\frac{num}{den}'.replace("num", f'{v[0]}').replace("den", f'{v[1]}')

         

        x1.value = r"<font color = 'white'><font size='3px'> $$x : {} $$".format(

            ans[0])

        y1.value = r"<font color = 'white'><font size='3px'> $$y :  {} $$".format(

            ans[1])

        z1.value = r"<font color = 'white'><font size='3px'> $$z :  {} $$".format(

            ans[2])

        plc2 = f'Time taken: {f"{t:.13f}"} seconds'

        time1.value = r"<font color = 'white'><font size='3px'> {}  ".format(

            plc2)





dic1 = {'x1': 'x :', 'y1': 'y :', 'z1': 'z :', 'time1': "Time taken:"}

dic2 = {'eq1': 'EQ1: (', 'eq2': 'EQ2: (', 'eq3': 'EQ3: (',

        'xpl': 'x) + (', 'ypl': 'y) + (', 'zpl': 'z) ='}





for j in dic2:

    exec(

        f"""{j}=HTMLMath(value=r"<font color = 'white'><font size='3px'>{dic2[j]}")""")

items = [Text(layout=Layout(width='60px', height='30px'))

         for w in range(1, 13)]

row1 = HBox([eq1, items[0], xpl, items[1], ypl, items[2], zpl, items[9]])

row2 = HBox([eq2, items[3], xpl, items[4], ypl, items[5], zpl, items[10]])

row3 = HBox([eq3, items[6], xpl, items[7], ypl, items[8], zpl, items[11]])



display(VBox([row1, row2, row3]))

for i in range(12):

    exec(f'items[{i}].on_submit(solsys)')

for j in dic1:

    exec(

        f"""{j}=HTMLMath(value=r"<font color = 'white'><font size='3px'>{dic1[j]}")""")

    exec(f'display({j})')