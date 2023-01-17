def create_list(input_size):

    return list(range(input_size)[::-1])
def test_enumerate(input_list):

    for i, v in enumerate(input_list):

        pass



def test_index(input_list):

    for i in range(len(input_list)):

        input_list[i]





test_list = create_list(10**3)



%timeit test_enumerate(test_list)

%timeit test_index(test_list)





test_list = create_list(10**6)



%timeit test_enumerate(test_list)

%timeit test_index(test_list)
# Adding this group of tests to prevent compiler optimization of loops



def test_enumerate2(input_list):

    for i, v in enumerate(input_list):

        b = i + v



def test_index2(input_list):

    for i in range(len(input_list)):

        b = i + input_list[i]





test_list = create_list(10**3)



%timeit test_enumerate2(test_list)

%timeit test_index2(test_list)





test_list = create_list(10**6)



%timeit test_enumerate2(test_list)

%timeit test_index2(test_list)