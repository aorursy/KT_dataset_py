def split(s, sep):

    result = []

    start = 0

    pos = 0

    while pos + len(sep) < len(s) + 1:

        if s[pos: pos + len(sep)] == sep:

            result.append(s[start: pos])

            pos = pos + len(sep)

            start = pos

        else:

            pos += 1

    result.append(s[start:])

    return result
print(split('abbbb', 'bb'))

print(split('abbbc', 'bb'))
def atoi(s):

    neg = False

    if s[0] == '-':

        neg = True

        s = s[1: ]

    val = {c: i for i, c in enumerate('0123456789')}

    v = 0

    for c in s:

        v = v * 10 + val[c]

    if neg:

        v *= -1

    return v
print(atoi('0'))

print(atoi('333') + atoi('-21'))