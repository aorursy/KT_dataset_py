def reverse_function():

    S = ArrayStack()

    original = S

    for line in S:

        S.push(line.rstrip('\n'))



    T = ArrayStack()

    output = T

    while not S.is_empty():

        output.write(S.pop() + '\n')
def queue_queue():

    D.dequeue = [1,2,3,4,5,6,7,8]

    for i in D:

        D.pop()



    Q = []

    while not D.is_empty():

        Q.enqueue(D)
def queue_stack():

    D.dequeue = [1,2,3,4,5,6,7,8]

    for i in D:

        D.pop()



    S = ArrayStack()

    while not D.is_empty():

        S.push(D)