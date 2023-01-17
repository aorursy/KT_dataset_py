import numpy as np



eps = 0.08

eps_2 = 0.08



def Func2(w,b,a):

    return (np.log(np.exp(w+b)/np.exp(a)))



def Func(w ,b ):

    return (np.log(1 / (1 + np.exp(w + b))))



def Diff_dw(w, b):

    return ((Func(w + eps, b) - Func(w, b)) / eps)



def Diff_dw2(w, b,a):

    return (((Func2(w + eps, b,a) - Func2(w, b,a)) / eps))



def Diff_db(w, b):

    return ((Func(w, b + eps) - Func(w, b)) / eps)



def Diff_db2(w, b, a):

    return ((Func2(w, b + eps,a) - Func2(w, b, a)) / eps)





def Gradient(w, b):

    return np.sqrt((w ** 2) + (b ** 2))



def GradietnDescent2(x, w, b, a, h):

    counter = 0

    w = w*x

    base_w = w

    base_b = b

    while (counter < 10):

        dw = Diff_dw2(w, b, a)

        db = Diff_db2(w, b, a)

        gradient = Gradient(dw, db)

        if (abs(gradient) < eps):

            return Func2(w, b, a)

        else:

            w = w - h * dw

            b = b - h * db



        if (Func2(w, b,a) - Func2(base_w, base_b,a) < 0):

            if (((abs(w - base_w) and abs(b - base_b)) < eps_2) and (

                    Func2(w, b,a) - Func2(base_w, base_b,a))< eps_2):

                return Func2(w, b, a)

        else:

            h = h / 2

            w = w - h * dw

            b = b - h * db

        counter += 1



    return Func2(w, b, a)



def GradientDescent(x, w, b,  h):

    counter = 0

    w = w*x

    base_w = w

    base_b = b

    while (counter < 10):

        dw = Diff_dw(w, b)

        db = Diff_db(w, b)

        gradient = Gradient(dw, db)

        if (abs(gradient) < eps):

            break

        else:

            w = w - h * dw

            b = b - h * db



        if (Func(w, b) - Func(base_w, base_b) < 0):

            if (((abs(w - base_w) and abs(b - base_b)) < eps_2) and (

                    Func(w, b) - Func(base_w, base_b))< eps_2):

                print('final: ', w, b, Func(w, b))

        else:

            h = h / 2

            w = w - h * dw

            b = b - h * db

        counter += 1



    return Func(w, b)



def binary_logistic_regression():

    w = np.linspace(0,1,10)

    w = np.resize(w, (w.size,1))



    x = np.array([5,6,7,8,9,5,6,7,8,9])

    h = 0.1

    b = 6

    q = []

    counter = 0

    for i in range(w.size):

        counter+=1

        q.append(GradientDescent( x[i],w[i], b,  h))



    print('q',q)

    print(-1/w.size*np.sum(q))



def nonbinary_logistic_regression():

    x = np.array([8, 8, 9])

    x = np.resize(x,(x.size,1))

    b = np.array([6,1,2,3])

    b = np.resize(b,(b.size,1))

    w = np.linspace(0,1,(x.size*b.size)).reshape(x.size,b.size)



    q = []

    d = []

    h = 0.1

    for j in range(b.size):

        for i in range(x.size):

            d.append(x[i]*w[i][j] + b[j])

    a = np.array(d)

    a = np.resize(a,(x.size,b.size))

    for j in range(b.size):

        for i in range(x.size):

            q.append(GradietnDescent2(x[i], w[i][j], b[j], a[i][j], h))

        print(q)

        print(-1/len(q)*(np.sum(q)))

        

def main():

    print("Binary logistic regression:")

    binary_logistic_regression()

    print()

    

    print("Nonbinary logistic regression:")

    nonbinary_logistic_regression()



if __name__ == '__main__':

    main()   