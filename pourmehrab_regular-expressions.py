from datetime import datetime
print(datetime.now())
print(f"Today's date is {datetime.now():%B %d, %Y}")
family = {"dad": "John", "siblings": "Peter"}
print(f"Is your dad called {family['dad']}?")
import re
# Validate the following string
password = "password1234"
re.search(r"\w{8}\d{4}", password)
# Once or more: +
text = "Date of start: 4-3. Date of registration: 10-04."
re.findall(r"\d+-\d+", text)
# Zero times or more: *
my_string = "The concert was amazing! @ameli!a @joh&&n @mary90"
re.findall(r"@\w+\W*\w+", my_string)
# Zero times or once: ?
text = "The color of this image is amazing. However, the colour blue could be brighter."
re.findall(r"colou?r", text)
# n times at least, m times at most : {n, m}
phone_number = "John: 1-966-847-3131 Michelle: 54-908-42-42424"
re.findall(r"\d{1,2}-\d{3}-\d{2,3}-\d{4,}", phone_number)
# Match any character (except newline): .
my_links = "Just check out this link: www.amazingpics.com. It has amazing photos!"
re.findall(r"www.+com", my_links)
# Start of the string: ^
my_string = "the 80s music was much better that the 90s"
re.findall(r"^the\s\d+s", my_string)
# End of the string: $
re.findall(r"the\s\d+s$", my_string)
# Escape special characters: \
my_string = "I love the music of Mr.Go. However, the sound was too loud."
print(re.split(r"\.\s", my_string))
# OR operator |
my_string = "Elephants are the world's largest land animal! I would love to see an elephant one day"
re.findall(r"Elephant|elephant", my_string)
# Set of characters: [ ]
# ^ transforms the expression to negative
my_links = "Bad website: www.99.com. Favorite site: www.hola.com"
re.findall(r"www[^0-9]+com", my_links)
