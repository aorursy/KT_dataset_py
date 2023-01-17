! pip install numerizer
from numerizer import numerize



print('seven million two thousand four', numerize('seven million two thousand four'))

print('twenty twenty', numerize('twenty twenty'))

print('two and three quarter', numerize('two and three quarter'))

print('one billion', numerize('one billion'))
! pip install faker
from faker import Faker



fake = Faker()



print('fake name:', fake.name())

print('fake first_name_female:', fake.first_name_female())

print('fake user_name:', fake.user_name())

print('fake password:', fake.password())

print('fake month:', fake.month())
! pip install emot
import emot



text = ';) :-) OMG :('

ans = emot.emoticons(text)



print(ans)

print()

print('value', ans['value'])

print('location', ans['location'])

print('mean', ans['mean'])

print('flag', ans['flag'])
! pip install pendulum
import pendulum



now_in_paris = pendulum.now('Europe/Paris')

print('Now in Paris', now_in_paris)

print('UTC', now_in_paris.in_timezone('UTC'))



print('Tomorrow', pendulum.now().add(days=1))



last_week = pendulum.now().subtract(weeks=1)

print('Last Week', last_week)



past = pendulum.now().subtract(minutes=2)

print('Past', past)

print('Diff for humans', past.diff_for_humans())



diff = past - last_week

print('Delta hours', diff.hours)

print('Delta words', diff.in_words(locale='en'))