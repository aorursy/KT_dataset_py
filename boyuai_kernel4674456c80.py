fp = open('../input/score.txt', 'r')

score = fp.readlines()
for i in range(0, len(score)):

    string = score[i].split()

    tot = 0

    for j in range(1, len(string)):

        tot = tot + int(string[j])

    print(string[0], ':', tot)