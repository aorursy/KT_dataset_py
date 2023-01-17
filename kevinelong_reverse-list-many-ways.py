# Q. How many ways are there to reverse a list in python?
# A. An infinite number but lets look at a few,
#     and weigh in on their pros and cons.

import time


# long but uses only tools beginners know
def by_hand_duplicate(data):
    output = []
    count = len(data) - 1
    while count >= 0:
        output.append(data[count])
        count -= 1
    return output


# faster, but requires deeper python knowledge and replaces list
def by_hand_in_place(data):
    half = len(data) // 2
    for i in range(half):
        data[i], data[-(1 + i)] = data[-(1 + i)], data[i]
    return data


# clear and fastest if not required to create a duplicate list
def reverse_d(data):
    return list(reversed(data))


#  This is the most concise, but requires advanced knowledge of
#  Python's peculiar slicing feature to even know what is going on.
def concise(data):
    return data[::-1]


# fastest but replaces the existing list
def reverse(data):
    data.reverse()
    return data


# TEST

# Let's do some pseudo-scientific profiling,
# and see how long it takes to run each one;
# a million times!

function_dict: {str: object} = {
    "by_hand_duplicate": by_hand_duplicate,
    "by_hand_in_place": by_hand_in_place,
    "reverse_d": reverse_d,
    "concise": concise,
    "reverse": reverse,
}


def test_reverse(f, data, iterations):
    begin_time = time.time()
    for i in range(iterations):
        f(data)
    return time.time() - begin_time


for name, f in function_dict.items():
    print(f"{name.upper()}:", end="")
    goal_data = ["cherry", "banana", "apple", "Z"]
    test_data = ["Z", "apple", "banana", "cherry"]
    result = f(test_data)
    assert result == goal_data
    print(test_reverse(f, test_data, 1_000_000))

"""
EXAMPLE OUTPUT:
BY_HAND_DUPLICATE:2.4218082427978516
BY_HAND_IN_PLACE:2.1684727668762207
REVERSE_D:1.233830213546753
CONCISE:0.9667091369628906
REVERSE:0.9249958992004395
"""
# As we can see there is a noticeable but likely unimportant
# performance difference even at a million iterations.
# Use the version that best fits your needs.
oneThroughFive = range(1,6)
# up to but not including the number on the right.
# [1,2,3,4,5]

zeroToFour = range(0,5)
# Up to but not including the number on the right.
#  [0,1,2,3,4]
print(list(oneThroughFive))
print(list(zeroToFour))


