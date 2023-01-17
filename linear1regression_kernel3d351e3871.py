import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
def my_cos(x,e=1e-10):

    answer = np.zeros(len(x)) + 1

    k = 0

    x_2 = x*x

    last = np.ones(len(x_2))

    while True:

        last *= -x_2/(2*k+1)/(2*k+2)

        k+=1

        if (abs(last)<e).all():

            break

        old = answer.copy()

        answer += last

    print("Итераций для косинуса:",k)

    print("Точность на предыдущей итерации для косинуса:",np.log10(max(abs(np.cos(x)-old))))



    return(answer)
x = np.array([1/4,1/2,3/4,0.66,4])

print(np.cos(x))

print(my_cos(x,1e-8))
def my_sin(x,e=1e-10):

    answer = x.copy()

    k = 0

    x_2 = x*x

    last = x.copy()

    while True:

        last *= -x_2/(2*k+2)/(2*k+3)

        k+=1

        if (abs(last)<e).all():

            break

        old = answer.copy()

        answer += last

    print("Итераций для синуса:",k)

    print("Точность на предыдущей итерации для синуса:",np.log10(max(abs(np.sin(x)-old))))

    return(answer)
print(my_sin(x))

print(np.sin(x))
def my_arctan(x,e=1e-5):

    index_s = (np.abs(x) < 1)

    bx = x[~index_s]

    sx = x[index_s]

    k1, k2 = 0,0

    answer = np.zeros(len(x))

    answer_s = sx.copy()

    sx_2 = sx*sx

    last = sx.copy()

    while True:

        last *= -sx_2*(2*k1+1)/(2*k1+3)





        if (abs(last)<e).all():

            break

            

        answer_s += last

        k1+=1



    bx_2 = 1/bx/bx

    last = 1/bx.copy()

    answer_b = 1/bx.copy()

    while True:

        last *= -bx_2*(2*(k2)+1)/(2*k2+3)

        if (abs(last)<e).all():

            break

        answer_b += last

        k2+=1

    print("Итераций для арктангенса:", max(k1,k2)+1)

    answer[index_s] = answer_s

    answer[~index_s] = np.sign(bx)*np.pi/2 - answer_b



    return(answer)
x = np.array([0.01,0.5,3/4,0.66,4000000])

print(np.arctan(x))

print(my_arctan(x,1e-4))
def my_sqrt(x,e=1e-5):

    ans_old = np.array([float("inf") for i in range(len(x))])

    ans = np.ones(len(x))

    k=0

    while (abs(ans_old - ans)>e).sum() != 0:

        k+=1

        old = ans_old

        ans_old = ans.copy()

        ans = 1/2*(ans + x/ans)

    print("Итераций для корня:",k)

    print("Точность на предыдущей итерации для корня:",np.log10(max(abs(old-ans_old))))

    return ans
print(np.sqrt(x))

print(my_sqrt(x,1e-8))
x = np.linspace(0.1,0.2,100)
real_answer = np.cos(2.8*x+np.sqrt(1+x))*np.arctan(1.5*x+0.2)

print(real_answer)
e = 1e-10

e1,e2,e3 = e/3/0.25,e/3/0.47,e/3

my_answer = -my_sin(2.8*x+my_sqrt(1+x,e1) - np.pi/2,e3)*my_arctan(1.5*x+0.2,e2)

print(my_answer)

e1,e2,e3
import plotly.graph_objects as go



fig = go.Figure()



fig.add_trace(go.Scatter(x=x, y=np.log10(abs(my_answer-real_answer)), name='Ox',

                         line=dict(color='firebrick',width=1.3)))

annotations = []

annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,

                              xanchor='left', yanchor='bottom',

                              text='Вся погрешность',

                              font=dict(family='Arial',

                                        size=30,

                                        color='rgb(37,37,37)'),

                              showarrow=False))



fig.update_layout(annotations=annotations)



fig.show()
import plotly.graph_objects as go



fig = go.Figure()



fig.add_trace(go.Scatter(x=x, y=np.log10(abs(np.cos(x)-my_cos(x,1e-6))), name='Ox',

                         line=dict(color='firebrick',width=1.4)))

annotations = []

annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,

                              xanchor='left', yanchor='bottom',

                              text='Cos(x)',

                              font=dict(family='Arial',

                                        size=30,

                                        color='rgb(37,37,37)'),

                              showarrow=False))



fig.update_layout(annotations=annotations)



fig.show()
import plotly.graph_objects as go



fig = go.Figure()

fig.add_trace(go.Scatter(x=x, y=(abs(my_sqrt(1+x,1e-4)-np.sqrt(1+x))), name='Ox',

                         line=dict(color='firebrick',width=1.3)))

annotations = []

annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,

                              xanchor='left', yanchor='bottom',

                              text='Sqrt(x)',

                              font=dict(family='Arial',

                                        size=30,

                                        color='rgb(37,37,37)'),

                              showarrow=False))



fig.update_layout(annotations=annotations)



fig.show()
import plotly.graph_objects as go



fig = go.Figure()



fig.add_trace(go.Scatter(x=x, y=np.log10(abs(np.arctan(1.5*x+0.2)-my_arctan(1.5*x+0.2,1e-6))), name='Ox',

                         line=dict(color='firebrick',width=1.3)))

annotations = []

annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,

                              xanchor='left', yanchor='bottom',

                              text='Arctan(x)',

                              font=dict(family='Arial',

                                        size=30,

                                        color='rgb(37,37,37)'),

                              showarrow=False))



fig.update_layout(annotations=annotations)



fig.show()
import plotly.graph_objects as go



fig = go.Figure()



fig.add_trace(go.Scatter(x=x, y=np.log10(abs(np.cos(2.8*x+np.sqrt(1+x))+my_sin(2.8*x+my_sqrt(1+x) - np.pi/2,3e-7))), name='Ox',

                         line=dict(color='firebrick',width=1.4)))

annotations = []

annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,

                              xanchor='left', yanchor='bottom',

                              text='cos(y) - my_cos(y); y = 2.8*x+np.sqrt(1+x)',

                              font=dict(family='Arial',

                                        size=30,

                                        color='rgb(37,37,37)'),

                              showarrow=False))



fig.update_layout(annotations=annotations)



fig.show()
abs(2.8*x+np.sqrt(1+x) - np.pi/2)<np.pi/4