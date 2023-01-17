def Create_Array(variable):

    n = 2**variable # Số hàng

    m = variable # Số cột ban đầu

    flag = 1

    array = [[0 for i in range(m)] for j in range(n)]

    for j in range(m):

        count = 1

        tmp = 1

        for i in range(n):

            if count == flag+1:

                array[i][j] = 1

                tmp += 1

                if (tmp == count):

                    tmp = 1

                    count = 1

                continue

            count += 1

        flag *= 2

    return array
def spilt(s):

    b = []

    n = []

    temp =''

    trigger = 0

    for i in s:

        if i == '(': trigger +=1

        elif i == ')': trigger -=1

        if trigger == 0:

            if i ==")":

                b.append(temp)

                temp = ''

            elif i == '-':

                n.append(len(b))

            else:

                b.append(i)

        else:

            if trigger == 1 and i=='(':

                continue

            else:

                temp += i

    return b, n

def spilt_block(s):

    block = []

    negative = []

    block.append(s)

    count = len(block)

    i = 0

    while i < count:

        if len(block[i]) != 1:

            b = []

            n = []

            b, n = spilt(block[i])

            n = [i +len(block) for i in n]

            block[i] = "{}:{}".format(len(block),len(block)+len(b)-1)

            block.extend(b)

            negative.extend(n)

        count = len(block)

        i += 1

    return block, negative

def binary_operation(a, operator, b):

    if operator == '*':

        return a and b

    elif operator == '+':

        return a or b

    elif operator == '<':

        if a == b: return 1

        else: return 0

    elif operator == '>':

        if a == 1 and b == 0: return 0

        else: return 1

    elif operator == '^':

        if a != b: return 1

        else: return 0

    elif operator == '':

        return a

    else: return "Unknown Operator"

def calculate_operation(element_list, operator_list):

    result = element_list[0]

    temp = 0

    for e in element_list[1:]:

        result = binary_operation(result,operator_list[temp],e)

        temp += 1

    return result

    
def calculate_block(blockk, negative, array):

    block = blockk.copy()

    dic = {}

    for i in range(26):

        dic[chr(65+i)] = i

    operator = ['*', '+', '>', '<', '^']

    i = len(block) - 1

    if i == 0:

        return array[dic[block[i]]]   

    while i >= 0:

        if len(block[i]) != 1:

            index_of_colon = block[i].index(':')

            x = ""

            for n in range(0,index_of_colon):

                x += block[i][n]

            y = ''

            for n in range(index_of_colon+1,len(block[i])):

                y += block[i][n]

            start = int(x)

            end = int(y)

            operator_list = []

            element_list = []

            for flag, op in enumerate(block[start:end+1]):

                if op in operator:

                    operator_list.append(op)

                else:

                    if op in dic:

                        temp = array[dic[op]]

                    else:

                        temp = int(op)

                    if flag+start in negative:

                        temp = abs(1-temp)

                    element_list.append(temp)

            result = calculate_operation(element_list, operator_list)

            display = result

            block[i] = result

        i -= 1

    return block[0]

def Cau_A(s, n, output):

    s = s.replace('<->', '<')

    s = s.replace('->', '>')

    block, negative = spilt_block(s)

    result = []

    for i in array:

        #print(block)

        #print(negative)

        temp = calculate_block(block, negative, i)

        #print(temp)

        result.append(temp)

    if output == True:

        return result

    else:        

        print('Biểu thức vừa nhập là: {}'.format(s))

        print('Danh sách các phep tinh cần phải tính: {}\nDanh sách vị trí các phần tử âm: {}'.format(block, negative))

        print('Giá trị của các biến: {}'.format(array))

        print('Kết quả cuối cùng là: {}'.format(result))

    
def Cau_BC(f, typee): # Tuyen chuan typee = 1, Hoi chuan typee = 0

    cid = {}

    for i in range(26):

        cid[i] = chr(65+i)

    result = ''

    openn = '('

    closee = ')'

    if typee == 1:

        x = '.'

        y = ' + '

        lenn = 0

        clas = 'C'

    else:

        x = '+'

        y = ' . '

        

        lenn = 2**n-1

        clas = 'D'

    for i, e in enumerate(f):

        if e == typee:

            small = ''

            j = abs(i - lenn)

            print('Vị trí: {}{}'.format(clas, i+1))

            for index, element in enumerate(array[j]):

                neg = ''

                if element == 0: neg = "'"

                small += x + cid[index] + neg

            small = small[1:]

            small = openn + small + closee

            result += y + small

    result = result[3:]

    return result

    

            

                

            
s = '(A->B)+C'

n = 3 # Số biến

array = Create_Array(n) # Tạo mảng 2 chiều

f = Cau_A(s, n, True) # Tinh #f

print('A) #f là {}'.format(f))

print('B) #f có dạng tuyển chuẩn là: {}'.format(Cau_BC(f, 1)))

print('C) #f có dạng hội chuẩn là: {}'.format(Cau_BC(f, 0)))
s = '((A->B)+C)*(D+E)'

n = 5 # Số biến

array = Create_Array(n) # Tạo mảng 2 chiều

f = Cau_A(s, n, True) # Tinh #f

print('A) #f là {}'.format(f))

print('B) #f có dạng tuyển chuẩn là: {}'.format(Cau_BC(f, 1)))

print('C) #f có dạng hội chuẩn là: {}'.format(Cau_BC(f, 0)))
s = '(A*B)+-B'

n = 2

array = Create_Array(n)

f = Cau_A(s, n, True)

print('A) #f là {}'.format(f))

print('B) #f có dạng tuyển chuẩn là: {}'.format(Cau_BC(f, 1)))

print('C) #f có dạng hội chuẩn là: {}'.format(Cau_BC(f, 0)))
s = 'A->B'

n = 2

array = Create_Array(n)

f = Cau_A(s, n, True)

print('A) #f là {}'.format(f))

print('B) #f có dạng tuyển chuẩn là: {}'.format(Cau_BC(f, 1)))

print('C) #f có dạng hội chuẩn là: {}'.format(Cau_BC(f, 0)))
s = '(-C+A+B)+A+B+C+C+(A+B)'

n = 3

array = Create_Array(n)

f = Cau_A(s, n, True)

print('A) #f là {}'.format(f))

print('B) #f có dạng tuyển chuẩn là: {}'.format(Cau_BC(f, 1)))

print('C) #f có dạng hội chuẩn là: {}'.format(Cau_BC(f, 0)))
s = '(A*B)+(A*C)+B'

n = 3

array = Create_Array(n)

f = Cau_A(s, n, True)

print('A) #f là {}'.format(f))

print('B) #f có dạng tuyển chuẩn là: {}'.format(Cau_BC(f, 1)))

print('C) #f có dạng hội chuẩn là: {}'.format(Cau_BC(f, 0)))
s = 'A'

n = 1

array = Create_Array(n)

f = Cau_A(s, n, True)

print('A) #f là {}'.format(f))

print('B) #f có dạng tuyển chuẩn là: {}'.format(Cau_BC(f, 1)))

print('C) #f có dạng hội chuẩn là: {}'.format(Cau_BC(f, 0)))