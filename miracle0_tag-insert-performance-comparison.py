from operator import itemgetter

from io import StringIO

from random import sample, seed



def tag_insert_no_op(text, tag_list):

    """Used to measure common operation times of all methods"""

    tag_list.sort(key=itemgetter(0))

    for offset, tag in tag_list:

        pass

    return text[:]



def tag_insert_concat1(text, tag_list):

    tag_list.sort(key=itemgetter(0), reverse=True)

    last_offset = len(text)

    output = ""

    for offset, tag in tag_list:

        output = tag + text[offset:last_offset] + output

        last_offset = offset

    return text[:last_offset] + output



def tag_insert_concat2(text, tag_list):

    tag_list.sort(key=itemgetter(0), reverse=True)

    for offset, tag in tag_list:

        text = text[:offset] + tag + text[offset:]

    return text



def tag_insert_concat3(text, tag_list):

    tag_list.sort(key=itemgetter(0), reverse=True)

    for offset, tag in tag_list:

        text = f"{text[:offset]}{tag}{text[offset:]}"

    return text



def tag_insert_concat4(text, tag_list):

    tag_list.sort(key=itemgetter(0))

    last_offset = 0

    output = ""

    for offset, tag in tag_list:

        output = output + text[last_offset:offset] + tag

        last_offset = offset

    return output + text[last_offset:]



def tag_insert_concat4_inp(text, tag_list):

    # Inplace version of tag_insert_concat4

    tag_list.sort(key=itemgetter(0))

    last_offset = 0

    output = ""

    for offset, tag in tag_list:

        output += text[last_offset:offset] + tag

        last_offset = offset

    return output + text[last_offset:]



def tag_insert_join1(text, tag_list):

    tag_list.sort(key=itemgetter(0))

    segs_to_join = []

    last_offset = 0

    for offset, tag in tag_list:

        segs_to_join.append(text[last_offset:offset])

        segs_to_join.append(tag)

        last_offset = offset

    segs_to_join.append(text[last_offset:])

    return ''.join(segs_to_join)



def tag_insert_join1_local(text, tag_list):

    tag_list.sort(key=itemgetter(0))

    segs_to_join = []

    last_offset = 0

    

    # Store append as a local variable to avoid class attribute lookup overhead

    append = segs_to_join.append

    for offset, tag in tag_list:

        append(text[last_offset:offset])

        append(tag)

        last_offset = offset

    append(text[last_offset:])

    return ''.join(segs_to_join)



def tag_insert_join1_extend(text, tag_list):

    tag_list.sort(key=itemgetter(0))

    segs_to_join = []

    last_offset = 0

    for offset, tag in tag_list:

        # Use += operator for list extension

        segs_to_join += (text[last_offset:offset], tag)

        last_offset = offset

    segs_to_join += (text[last_offset:],)

    return ''.join(segs_to_join)



def tag_insert_join1_yield(text, tag_list):

    tag_list.sort(key=itemgetter(0))

    def generate_segs():

        last_offset = 0

        for offset, tag in tag_list:

            yield text[last_offset:offset]

            yield tag

            last_offset = offset

        yield text[last_offset:]



    # Unpack generator to list for string joining,

    # see https://stackoverflow.com/questions/9060653/list-comprehension-without-in-python/9061024#9061024

    return ''.join([*generate_segs()])



def tag_insert_join1_yield2(text, tag_list):

    tag_list.sort(key=itemgetter(0))

    def generate_segs():

        last_offset = 0

        for offset, tag in tag_list:

            yield text[last_offset:offset]

            yield tag

            last_offset = offset

        yield text[last_offset:]

    return ''.join(generate_segs())  # Use generator directly for joining



def tag_insert_join2(text, tag_list):

    tag_list.sort(key=itemgetter(0))

    segs_generator = (tag + text[offset:offset2] for (offset, tag), (offset2, _) in zip(tag_list, tag_list[1:]))

    last_offset, last_tag = tag_list[-1]

    segs_to_join = [text[:tag_list[0][0]], *segs_generator, last_tag, text[last_offset:]]

    return ''.join(segs_to_join)



def tag_insert_join3(text, tag_list):

    tag_list.sort(key=itemgetter(0))

    def segs_generator():

        for (offset, tag), (offset2, _) in zip(tag_list, tag_list[1:]):

            yield tag

            yield text[offset:offset2]

    last_offset, last_tag = tag_list[-1]

    segs_to_join = [text[:tag_list[0][0]], *segs_generator(), last_tag, text[last_offset:]]

    return ''.join(segs_to_join)

    

def tag_insert_join3_iter(text, tag_list):

    tag_list.sort(key=itemgetter(0))

    def segs_generator():

        it = iter(tag_list)

        yield text[:next(it)[0]]

        for (offset, tag), (offset2, _) in zip(tag_list, it):

            yield tag

            yield text[offset:offset2]

    last_offset, last_tag = tag_list[-1]

    segs_to_join = [*segs_generator(), last_tag, text[last_offset:]]

    return ''.join(segs_to_join)



def tag_insert_format(text, tag_list):

    tag_list.sort(key=itemgetter(0))

    offsets, tags = zip(*tag_list)

    text_segs = [text[:offsets[0]]] 

    text_segs += (text[prev_offset:offset] for prev_offset, offset in zip(offsets, offsets[1:]))

    text_segs += (text[offsets[-1]:], )

    fmt_str = "{}".join(text_segs)

    return fmt_str.format(*tags)



def tag_insert_bytearray(text, tag_list):

    tag_list.sort(key=itemgetter(0))

    output = bytearray(len(text) + len(tag_list) * 4)

    last_offset = 0

    output_offset = 0

    len_func = len

    encode = str.encode

    text = encode(text)

    for offset, tag in tag_list:

        new_offset = output_offset + offset - last_offset

        output[output_offset:new_offset] = text[last_offset:offset]

        output_offset, new_offset = new_offset, new_offset + len_func(tag)

        output[output_offset:new_offset] = encode(tag)

        output_offset = new_offset

        last_offset = offset

    output[output_offset:] = text[last_offset:]

    return output.decode()



def tag_insert_string_io(text, tag_list):

    tag_list.sort(key=itemgetter(0))

    out = StringIO()

    write = out.write

    

    last_offset = 0

    for offset, tag in tag_list:

        write(text[last_offset:offset])

        write(tag)

        last_offset = offset

    write(text[last_offset:])

    return out.getvalue()



def tag_insert_test(func, test_cases):

    for text, tag_list, exp in test_cases:

        out = func(text, tag_list)

        assert out == exp, f"\n{func.__name__}  Input:    {text}\nOutput:   {out}\nExpected: {exp}\n"



def generate_test_case(text_len):

    text = "abdce" * (text_len // 5)

    tag_list = sample([*enumerate(text.upper(), 1)], len(text))

    expected = ''.join(c+c.upper() for c in text)

    return text, tag_list, expected

    

def run():

    from timeit import timeit

    from platform import python_implementation

    import sys



    print(sys.version)

    print(python_implementation())



    test_cases = [

        ("abcde", [(2, 'A'), (4, 'B'), (5, 'C')], "abAcdBeC"),

        ("edcba", [(1, 'X'), (2, 'Z'), (3, 'Z')], "eXdZcZba"),

    ]



    seed(0)

    text_tag_lengths_to_test = [150, 300, 450, 600, 750, 900]

    perf_test_cases = [*map(generate_test_case, text_tag_lengths_to_test)]

    test_cases.extend(perf_test_cases)



    funcs_to_run = (

        tag_insert_concat1, tag_insert_concat2, tag_insert_concat3,

        tag_insert_concat4, tag_insert_concat4_inp,

        tag_insert_join1, tag_insert_join1_local, tag_insert_join1_extend,

        tag_insert_join1_yield, tag_insert_join1_yield2,

        tag_insert_join2, tag_insert_join3, tag_insert_join3_iter,

        tag_insert_format, tag_insert_string_io

    )

    

    for func in funcs_to_run:

        tag_insert_test(func, test_cases)

    

    num_runs = 10**4

    cell_len = 8

    

    print()

    print(f"{'Mean exec time (us)':^24}{'Text Length':^{cell_len * len(text_tag_lengths_to_test)}}")

    print(f"{'Method':<24}", *(f"{l:^{cell_len}}" for l in text_tag_lengths_to_test), sep='')

    for func in (tag_insert_no_op,) + funcs_to_run:

        print(end=f"{func.__name__:<24}")

        for text, tag_list, _ in perf_test_cases:

            measure = timeit(lambda: func(text, tag_list), number=num_runs)

            print(end=f"{measure * 10**6 / num_runs:^{cell_len}.2f}")

        print()



if __name__ == "__main__":

    run()