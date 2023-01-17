def write_list_to_file(text_list, file_name):
    output_data = "\n".join(text_list)
    output_file = open(file_name, "w")
    output_file.write(output_data)
    output_file.close()


def read_list_from_file(file_name):
    output_word_list = []    
    input_file = open(file_name, "r")
    line_list = input_file.readlines()
    input_file.close()
    for word in line_list:
        output_word_list.append(word.strip())
    return output_word_list

word_list = ["Larry", "Moe", "Curly"]
print(word_list)
write_list_to_file(word_list, "data.txt")


# --------------
result_word_list = read_list_from_file("data.txt")
print(result_word_list)
assert word_list == result_word_list

output_file = open("example.txt", "w")
output_file.write("""
Now is the time,
for all good people,
to come to the aid,
of their planet.
""")
output_file.close()
