import re

phoneNumRegex=re.compile(r'\d\d\d-\d\d\d-\d\d\d\d')
print(phoneNumRegex.findall('Call me 415-555-2211 tomorrow, or at 415-333-6622 in my office'))