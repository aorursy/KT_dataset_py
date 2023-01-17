def Create_Parameter_Array(variable,root):

    n = 2**variable # Số hàng

    m = variable+root # Số cột ban đầu

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
def Update_Parameter_Array(number, n_bien, n_an):

    row = (2**n_bien)

    bit_length = row*n_an

    bit = '{0:0' + str(bit_length) + 'b}'

    start = n_bien

    end = n_bien + n_an

    string_bit = bit.format(number)

    root = []

    temp = []

    flag = 0

    for s in string_bit:

        flag += 1

        t = int(s)

        temp.append(t)

        if flag == row:

            another_temp = temp.copy()

            root.append(another_temp)            

            flag = 0

            temp.clear()

    for col in range(start,end):

        for i in range(0,row):

            array[i][col] = root[col-start][i]

    return root

    
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

    for i in range(n_bien):

        dic[chr(65+i)] = i

    for j in list_an:

        i += 1

        dic[j] = i

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
def In_Nghiem(f,n_bien):

    cid = {}

    for i in range(26):

        cid[i] = chr(65+i)

    result = ''

    openn = '('

    closee = ')'

    x = '.'

    y = ' + '

    lenn = 0

    clas = 'C'

    for i, e in enumerate(f):

        if e == 1:

            small = ''

            j = abs(i - lenn)

            #print('Vị trí: {}{}'.format(clas, i+1))

            for index, element in enumerate(array[j]):

                if index >= n_bien:

                    break

                neg = ''

                if element == 0: neg = "'"

                small += x + cid[index] + neg

            small = small[1:]

            small = openn + small + closee

            result += y + small

    result = result[3:]

    return result
def Tim_NghiemPT(s, n, showing_all):

    s = s.replace('<->', '<')

    s = s.replace('->', '>')

    left_right = s.split(' = ',1)

    left = left_right[0]

    right = left_right[1]

    left_block, left_negative = spilt_block(left)

    right_block, right_negative = spilt_block(right)

    case = 2**(2**n)

    case **= len(list_an)

    for index in range(1,case):

        root = Update_Parameter_Array(index,n,len(list_an))

        empty = 0

        #print(root)

        for r in root:

            if 1 in r: 

                empty += 1

        #print(empty)        

        if empty < len(list_an): continue

        left_r = []

        right_r = []

        for i in array:

            #print(block)

            #print(negative)

            left_temp = calculate_block(left_block, left_negative, i)

            right_temp = calculate_block(right_block, right_negative, i)

            #print(temp)

            left_r.append(left_temp)

            right_r.append(right_temp)

        nghiem = []

        if left_r == right_r:

            count = 0

            for r in root:

                print('Với trường hợp ẩn {} = {}'.format(list_an[count],root[count]))

                nghiem.append(In_Nghiem(root[count],n))

                count += 1

            print('Trường hợp này có nghiệm do hai vế bằng nhau với Vế trái = {} và Vế phải = {}'.format(left_r,right_r))

            print('Vậy nghiệm của phương trình là: ')

            for r in range(len(list_an)):

                print('{} = {}'.format(list_an[r],nghiem[r]))

            print('\n')

        else:

            if showing_all:

                count = 0

                for r in root:

                    print('Với trường hợp ẩn {} = {}'.format(list_an[count],root[count]))

                    count += 1

                print('Trường hợp này không có nghiệm, do hai vế không bằng nhau với Vế trái = {} và Vế phải = {}\n'.format(left_r,right_r))

        #print('#f có giá trị là: {} \n'.format(r))

    

    
string = '(B+A)*X = B+A'

n_bien = 2 # Số biến

list_an = ['X'] #Danh sách các ẩn trong phương trình

array = Create_Parameter_Array(n_bien,len(list_an)) # Tạo mảng 2 chiều

Tim_NghiemPT(string, n_bien, True) # True nếu muốn in cả trường hợp vô nghiệm

string = 'A*X = A'

n_bien = 2 # Số biến

list_an = ['X'] #Danh sách các ẩn trong phương trình

array = Create_Parameter_Array(n_bien,len(list_an)) # Tạo mảng 2 chiều

Tim_NghiemPT(string, n_bien, True) # True nếu muốn in cả trường hợp vô nghiệm
string = 'A*X = Y*X'

n_bien = 1 # Số biến

list_an = ['X','Y'] #Danh sách các ẩn trong phương trình

array = Create_Parameter_Array(n_bien,len(list_an)) # Tạo mảng 2 chiều

Tim_NghiemPT(string, n_bien, False) # True nếu muốn in cả trường hợp vô nghiệm
string = '(B+A)*X = (B+A+C)*Y+(A*X)'

n_bien = 3 # Số biến

list_an = ['X','Y'] #Danh sách các ẩn trong phương trình

array = Create_Parameter_Array(n_bien,len(list_an)) # Tạo mảng 2 chiều

Tim_NghiemPT(string, n_bien, False) # True nếu muốn in cả trường hợp vô nghiệm
string = 'X*(A+B)+(Y*B) = X*A'

n_bien = 2 # Số biến

list_an = ['X','Y'] #Danh sách các ẩn trong phương trình

array = Create_Parameter_Array(n_bien,len(list_an)) # Tạo mảng 2 chiều

Tim_NghiemPT(string, n_bien, False) # True nếu muốn in cả trường hợp vô nghiệm