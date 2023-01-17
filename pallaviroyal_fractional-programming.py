import numpy as np
# Simulator to simulate student marks in m exams. 

def studentMarksInExams(lower_limit_tm, upper_limt_tm, m):

    '''

    Params

        - lower_limit_tm : Lower limit of exams's total marks for simulation.

        - lower_limit_tm : upper limit of exam's total marks for simulation.

        - m : Number of exams conducted to the student.

    return

        - a : Student marks at each exam (m exams). Student gets minimum  35% and max 100% in each exam.

        - b : Total marks of each exam (m exams)

    '''

    # util

    def _get_random_numbers(ll, ul, size):

        return np.random.randint(ll, ul, size=(size))

    a = []

    b = []

    # If you would like to have each exam for 100 marks then un comment bellow like and comment next two lines.

#     b = [100] * m

    b = _get_random_numbers(lower_limit_tm, upper_limt_tm, m)

    b = list(b)

    for tm in b:

        # Student gets minimum  35% and max 100% in each exam.

        a.append(list(_get_random_numbers(int(0.35 * tm), tm, 1))[0])

    return np.array(a), np.array(b)
# As per Dinkelbach theorem. But need to find optimal "lda" to get student's best top n exams.

# We are going to find that using Golden section search-https://en.wikipedia.org/wiki/Golden-section_search. Implementation available in next function.



def top_n_exams(lda, a, b, n):

    m = len(a)

    values = []

    for i in range(m):

        values.append(a[i]-(lda * b[i]))

    #values.sort(reverse=True)

    return sum(values[-n:]), np.argsort(values)[-n:]
# Golden section search-https://en.wikipedia.org/wiki/Golden-section_search

def gss(m, n, a, b, l, r):

    for k in range(m):

        # Here 0.618 is the gss factor. More information go through above link.

        xl = r - 0.618*(r-l)

        xr = l + 0.618*(r - l)

        fl, tmp = top_n_exams(xl,a, b, n)

        fl = fl**2

        fr, tmp = top_n_exams(xr,a, b, n)

        fr = fr**2

        if(fl>=fr):

            l = xl

        else:

            r = xr

    # l,r function  boundary, 0.5*(l+r) can be minimum value of given function.

    return l, r

    

    
# Total number of exams

m = 10

# Choose top n exams which student scored good marks. 

n =5

# Students marks in m exams

a, b = studentMarksInExams(35, 100, m)
# here l - minimum and r - maximum are initial boundary which we assumed for GSS.

l = 0

r = np.sum(np.sort(a)[-n:])/np.sum(np.sort(b)[:n])

# min_boundary 

l, r= gss(m, n, a, b, l, r)
# This is something which we were looking for

f, subs = top_n_exams(0.5*(l+r), a, b, n)

print("Total number of exams m = ", m)

print("Consider for calulation best top n = ", n)

print("Student marks in each exam a = ",a)

print("Total marks of each exam b =   ", b)

print('The optimal subset consists of the ' + str(np.sort(subs)) + ' elements. This list is indexes of our a and b list. Index start with 0.')

print('The optimal value is: ' + str(np.sum(a[subs])/np.sum(b[subs])))

    