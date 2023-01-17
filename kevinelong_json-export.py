python_data = [
    { 'name':"kevin", "phone":"503-888-6879", "age": 52},
    { "name":"nina", "phone":"555-555-6879", "age": 43}
]

import json

text_output = json.dumps(python_data, indent=4)

print(len(text_output))
print(text_output)
f = open("data.json", "w") # w, r, a
f.write(text_output)
f.close()
f = open("data.json", "r") # w, r, a
raw_text = lines = f.read()
f.close()
print(raw_text)

data = json.loads(raw_text)
print(data)
# LOG
def write_log_item(text_output):
    f = open("log.txt", "a") # w, r, a
    f.write(text_output + "\n")
    f.close()

write_log_item("text_output 1")
write_log_item("text_output 2")
write_log_item("text_output 3")