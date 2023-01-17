import re
print('I would like some vegetables.'.replace('vegetables', 'pie'))

print(re.sub('vegetables', 'pie', 'I would like some vegetables.'))
veggie_request = 'I would like some vegetables, vitamins, and water.'

print(veggie_request.replace('vegetables', 'pie')

    .replace('vitamins', 'pie')

    .replace('water', 'pie'))

print(re.sub('vegetables|vitamins|water', 'pie', veggie_request))
messy_phone_number = '(123) 456-7890'

print(re.sub(r'\D', '', messy_phone_number))
really_messy_number = messy_phone_number + ' this is not a valid phone number'

print(re.sub(r'\D', '', really_messy_number))

print(re.sub(r'[-.() ]', '', really_messy_number))
buried_phone_number = 'You are the 987th caller in line for 1234567890. Please continue to hold.'

re.findall(r'\d{10}', buried_phone_number)
re.findall(r'\d{3}(?=\d{7})', buried_phone_number)
wordy_tom = """Tom. Let's talk about him. He often forgets to capitalize tom, his name. Oh, and don't match tomorrow."""

re.findall(r'(?i)\bTom\b', wordy_tom)