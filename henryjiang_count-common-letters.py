def count_common_letters_1(s1, s2):
    counter = 0
    for ch in set(s1.lower()):
        if ch in set(s2.lower()):
            counter += 1
    return counter

def count_common_letters_2(s1, s2):
    common = ''
    for ch in s1.lower():
        if ch in s2.lower() and ch not in common:
            common += ch
    return len(common)

if __name__ == '__main__':
    s = 'Bananac'
    t = 'CaN'
    print(count_common_letters_1(s,t))
    print(count_common_letters_2(s,t))
