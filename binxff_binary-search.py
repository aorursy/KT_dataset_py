!python3.7 -m pip install terminaltables
from terminaltables import AsciiTable

# ^^^ NOTE: use pip to install

# terminaltables to run locally



def main():

    _list = [12,13,18,19,20,31,35,39]

    key = 30



    print('List; Key ')

    print(_list, end='')

    print('; '+str(key))

    _lower = getInsetValue(_list, key)

    print('New List: ',end='')

    setInsertValue(_lower, _list, key)



def getInsetValue(_list:list, key:int):

    _table = []

    _lower = 0  # first value of a list

    _upper = len(_list)  # len of the list

    _table.append(  # Headders for Ascii table

        ['Lower i', 'Upper i', 'Test 1', 'Middle i', 'Test 2'] 

    )

    while _lower<_upper:

        _middle = (_lower+_upper)/2

        if _middle%2==1:

            _middle-=0.5  # Will round to the lower value

        _middle = int(_middle)  # converts to integer

        _table.append([

                _lower, _upper, _lower<_upper, _middle, key>_list[_middle]

            ])

        if key>_list[_middle]:

            _lower = _middle+1

        else:

            _upper = _middle

    _table.append([  # Last one; as its False it wont add it.

                _lower, _upper, _lower<_upper, _middle, key>_list[_middle]

    ])

    table = AsciiTable(_table)

    print(table.table)

    print("Test 1 = (Lower i)<(Upper i)")

    print("Test 2 = (Key)>(List[Middle i])")

    print("Insert Value: ", _lower)

    return _lower  # Returns index



def setInsertValue(_lower:int, _list:list, key:int):

    _list.append(_list[len(_list)-1])  # ? Does it matter what value it is?



    # -- An Example:

    # [1,3,4,5,6,7] + [7] <-- does it matter what the value if, woundent it just be copyed in the while loop

    # [1,3,4,5,6,7,7]

    # [1,3,4,5,6,6,7]

    # [1,3,4,5,5,6,7]

    # ...

    # [1,3,3,4,5,6,7]

    # [1,2,3,4,5,6,7]

    

    index = len(_list)-1

    while index > _lower:

        _list[index] = _list[index - 1]

        index = index - 1

    _list[_lower] = key

    print(_list)



if __name__ == "__main__":

    main()